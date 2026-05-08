import os
import requests
import base64
import numpy as np
import json
import copy
import concurrent.futures
import time
from io import BytesIO
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

# --- Helper Functions ---

MAX_INLINE_REQUEST_BYTES = 20 * 1024 * 1024
SAFE_INLINE_BUDGET_BYTES = 19 * 1024 * 1024

MODEL_CAPABILITIES = {
    "gemini-2.5-flash-image": {
        "supports_512": False,
        "supports_extended_ratios": False,
        "supports_google_search": False,
        "supports_thinking_toggle": False,
        "max_images": 3,
    },
    "gemini-3-pro-image-preview": {
        "supports_512": False,
        "supports_extended_ratios": False,
        "supports_google_search": True,
        "supports_thinking_toggle": False,
        "max_images": 14,
    },
    "gemini-3.1-flash-image-preview": {
        "supports_512": True,
        "supports_extended_ratios": True,
        "supports_google_search": True,
        "supports_thinking_toggle": True,
        "max_images": 14,
    },
}

BASE_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "5:4", "4:5", "21:9"}
EXTENDED_31_RATIOS = {"1:4", "4:1", "1:8", "8:1"}
ALL_ASPECT_RATIOS = sorted(BASE_ASPECT_RATIOS.union(EXTENDED_31_RATIOS))
OUTPUT_FORMAT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}

# USD per 1M tokens. The node uses standard Gemini Developer API pricing.
PRICING_USD_PER_1M = {
    "gemini-2.5-flash-image": {"input": 0.30, "output_text": 2.50, "output_image": 30.00},
    "gemini-3.1-flash-image-preview": {"input": 0.50, "output_text": 3.00, "output_image": 60.00},
    "gemini-3-pro-image-preview": {"input": 2.00, "output_text": 12.00, "output_image": 120.00},
}


def _validate_model_settings(model, aspect_ratio, resolution, use_google_search, enable_thinking):
    if model not in MODEL_CAPABILITIES:
        raise ValueError(f"Unsupported model '{model}'.")

    caps = MODEL_CAPABILITIES[model]
    valid_aspects = BASE_ASPECT_RATIOS.union(EXTENDED_31_RATIOS if caps["supports_extended_ratios"] else set())
    if aspect_ratio not in valid_aspects:
        supported = ", ".join(sorted(valid_aspects))
        raise ValueError(
            f"Aspect ratio '{aspect_ratio}' is not supported by model '{model}'. Supported values: {supported}"
        )

    if resolution == "512" and not caps["supports_512"]:
        raise ValueError(f"Resolution '512' (0.5K) is only supported by gemini-3.1-flash-image-preview.")

    if model == "gemini-2.5-flash-image" and resolution != "1K":
        raise ValueError("gemini-2.5-flash-image supports 1K output only in this node.")

    if use_google_search and not caps["supports_google_search"]:
        raise ValueError(f"Google Search tool is not supported by model '{model}'.")

    if enable_thinking and not caps["supports_thinking_toggle"]:
        print(f"NanoGemini Info: Thinking toggle is ignored for model '{model}'.")

    return caps


def _normalize_seed_for_api(seed):
    """
    Gemini expects generationConfig.seed as INT32.
    Comfy workflows often produce UINT64 seeds, so normalize safely.
    """
    if seed is None:
        return None

    max_i32 = 2147483647
    min_i32 = -2147483648
    s = int(seed)

    if min_i32 <= s <= max_i32:
        return s

    # Keep deterministic behavior while mapping into valid signed INT32 range.
    normalized = s % (max_i32 + 1)
    print(
        f"NanoGemini Warning: Seed {s} is outside INT32 range. "
        f"Using normalized seed {normalized} for Gemini API."
    )
    return normalized


def _extract_error_text(response):
    if response is None:
        return "No response details available."
    try:
        payload = response.json()
        return json.dumps(payload, indent=2)
    except Exception:
        return response.text


def _upload_file_via_api(file_bytes, mime_type, display_name, gemini_api_key, timeout=120):
    start_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={gemini_api_key}"
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    start_body = {"file": {"display_name": display_name}}

    start_response = requests.post(start_url, headers=start_headers, json=start_body, timeout=timeout)
    try:
        start_response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = _extract_error_text(start_response)
        raise ValueError(f"File API upload start failed (HTTP {start_response.status_code}): {detail}") from e

    upload_url = start_response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise ValueError("File API upload failed: missing resumable upload URL in response headers.")

    upload_headers = {
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
        "Content-Length": str(len(file_bytes)),
    }
    upload_response = requests.post(upload_url, headers=upload_headers, data=file_bytes, timeout=timeout)
    try:
        upload_response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = _extract_error_text(upload_response)
        raise ValueError(f"File API upload finalize failed (HTTP {upload_response.status_code}): {detail}") from e

    try:
        upload_data = upload_response.json()
    except Exception as e:
        raise ValueError("File API upload failed: response was not valid JSON.") from e

    file_uri = ((upload_data.get("file") or {}).get("uri"))
    if not file_uri:
        raise ValueError(f"File API upload failed: missing file URI in response: {json.dumps(upload_data, indent=2)}")

    return file_uri


def _normalize_output_image(pil_image, output_mime_type):
    if output_mime_type == "image/png":
        return pil_image.convert("RGBA" if "A" in pil_image.getbands() else "RGB")
    if output_mime_type == "image/jpeg":
        return pil_image.convert("RGB")
    if output_mime_type == "image/webp":
        return pil_image.convert("RGBA" if "A" in pil_image.getbands() else "RGB")
    raise ValueError(f"Unsupported output mime type '{output_mime_type}'.")


def _encode_output_preview(pil_image, output_mime_type):
    fmt_map = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/webp": "WEBP",
    }
    fmt = fmt_map[output_mime_type]
    buf = BytesIO()
    normalized = _normalize_output_image(pil_image, output_mime_type)
    save_kwargs = {"quality": 95} if fmt in {"JPEG", "WEBP"} else {}
    normalized.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    return Image.open(buf).copy()


def _tokens_by_modality(details):
    counts = {}
    if not isinstance(details, list):
        return counts

    for item in details:
        if not isinstance(item, dict):
            continue

        modality = str(item.get("modality", "UNSPECIFIED")).upper()
        token_count = item.get("tokenCount", 0) or 0

        try:
            token_count = int(token_count)
        except (TypeError, ValueError):
            token_count = 0

        counts[modality] = counts.get(modality, 0) + token_count

    return counts


def _estimate_gemini_cost(usage, model):
    rates = PRICING_USD_PER_1M.get(model)
    if not rates:
        return None

    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount") or 0)

    prompt_by_modality = _tokens_by_modality(usage.get("promptTokensDetails"))
    candidate_by_modality = _tokens_by_modality(usage.get("candidatesTokensDetails"))

    input_tokens = sum(prompt_by_modality.values()) if prompt_by_modality else prompt_tokens
    image_output_tokens = candidate_by_modality.get("IMAGE", 0)
    known_candidate_tokens = sum(candidate_by_modality.values())
    output_text_tokens = sum(
        count
        for modality, count in candidate_by_modality.items()
        if modality != "IMAGE"
    )

    if candidate_by_modality:
        unclassified_output_tokens = max(candidate_tokens - known_candidate_tokens, 0)
    else:
        unclassified_output_tokens = candidate_tokens

    return (
        input_tokens * rates["input"] / 1_000_000
        + (output_text_tokens + unclassified_output_tokens) * rates["output_text"] / 1_000_000
        + image_output_tokens * rates["output_image"] / 1_000_000
    )


def _print_gemini_usage_summary(model, usage):
    estimated_cost = _estimate_gemini_cost(usage, model)
    total_cost = "unavailable" if estimated_cost is None else f"${estimated_cost:.8f}"

    print(f"input token: {usage.get('promptTokenCount')}", flush=True)
    print(f"output token: {usage.get('candidatesTokenCount')}", flush=True)
    print(f"total token: {usage.get('totalTokenCount')}", flush=True)
    print(f"total cost: {total_cost}", flush=True)


def _log_gemini_usage(model, request_idx, usage, input_image_count):
    """
    Logs Gemini REST usageMetadata per generateContent call.
    Writes to output/gemini_usage.jsonl and prints to the ComfyUI terminal.
    """
    if not isinstance(usage, dict):
        print(
            f"[NanoGemini usage] req={request_idx} model={model} usageMetadata missing",
            flush=True,
        )
        return

    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request": request_idx,
        "model": model,
        "input_image_count": input_image_count,
        "promptTokenCount": usage.get("promptTokenCount"),
        "candidatesTokenCount": usage.get("candidatesTokenCount"),
        "totalTokenCount": usage.get("totalTokenCount"),
        "thoughtsTokenCount": usage.get("thoughtsTokenCount"),
        "toolUsePromptTokenCount": usage.get("toolUsePromptTokenCount"),
        "cachedContentTokenCount": usage.get("cachedContentTokenCount"),
        "promptTokensDetails": usage.get("promptTokensDetails"),
        "candidatesTokensDetails": usage.get("candidatesTokensDetails"),
        "toolUsePromptTokensDetails": usage.get("toolUsePromptTokensDetails"),
    }

    cost = _estimate_gemini_cost(usage, model)
    if cost is not None:
        row["estimatedCostUsd"] = cost

    line = json.dumps(row, ensure_ascii=False)
    _print_gemini_usage_summary(model, usage)

    try:
        out_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)

        log_path = os.path.join(out_dir, "gemini_usage.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[NanoGemini usage] failed to write usage log: {e}", flush=True)


def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B=1, H, W, C) to PIL Image (RGB)"""
    if image_tensor is None or image_tensor.shape[0] == 0:
        return None
    # Take the first image in the batch if a batch is passed to a single socket
    i = 255. * image_tensor[0].cpu().numpy()  # (H, W, C)
    image = np.clip(i, 0, 255).astype(np.uint8)
    
    c = image.shape[-1]
    if c == 1:
        image = np.repeat(image, 3, axis=-1)
    elif c == 3:
        pass
    elif c == 4:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported channels: {c}. Expected 1, 3, or 4.")
    
    return Image.fromarray(image, mode='RGB')

def pil2tensor(pil_image):
    """Convert PIL Image (RGB) back to ComfyUI tensor (B=1, H, W, C)"""
    if torch is None:
        raise RuntimeError("Torch is required to convert PIL images to ComfyUI tensors.")
    if pil_image is None:
        return None
    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    return torch.from_numpy(arr)

class NanoBEditGemini:
    """
    ComfyUI Node for Google Gemini Image Editing.
    Supports Gemini 3 Pro, Gemini 3.1 Flash Image, and Gemini 2.5 Flash Image.
    Directly hits the Google Generative Language API.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": ([
                    "gemini-3-pro-image-preview", # Nano Banana Pro
                    "gemini-3.1-flash-image-preview", # Nano Banana 2
                    "gemini-2.5-flash-image"      # Nano Banana
                ], {"default": "gemini-3-pro-image-preview"}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Added seed control
                "aspect_ratio": (ALL_ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "output_format": (["png", "jpg", "webp"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_file_api": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_google_search": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "safety_filter": (["block_none", "block_few", "block_some", "block_most"], {"default": "block_none"}),
                "debug_payload": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6_14": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "process"
    CATEGORY = "NanoGemini"
    OUTPUT_NODE = True

    def process(self, prompt, model, gemini_api_key, seed, aspect_ratio="1:1", resolution="1K", output_format="png",
                num_images=1, use_file_api=False, use_google_search=False, enable_thinking=False, safety_filter="block_none", debug_payload=False,
                image1=None, image2=None, image3=None, image4=None, image5=None, image6_14=None):
        
        # Prefer local environment variable by default, fallback to UI value.
        env_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        ui_api_key = (gemini_api_key or "").strip()

        if env_api_key:
            gemini_api_key = env_api_key
            if ui_api_key and ui_api_key != env_api_key:
                print("NanoGemini Info: Using GEMINI_API_KEY from environment (UI key ignored).")
        elif ui_api_key:
            gemini_api_key = ui_api_key
        else:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY in your environment or provide a key in the node UI."
            )

        if output_format not in OUTPUT_FORMAT_TO_MIME:
            raise ValueError(
                f"Unsupported output format '{output_format}'. Supported values: {', '.join(OUTPUT_FORMAT_TO_MIME.keys())}"
            )
        output_mime_type = OUTPUT_FORMAT_TO_MIME[output_format]

        # Keep UI label user-friendly while sending API-compatible value.
        api_resolution = "512" if resolution == "0.5K" else resolution

        caps = _validate_model_settings(model, aspect_ratio, api_resolution, use_google_search, enable_thinking)

        # 1. Prepare Input Images
        input_tensors = []
        
        # Add individual optional images
        for img in [image1, image2, image3, image4, image5]:
            if img is not None:
                input_tensors.append(img) # These are usually [1, H, W, C]
        
        # Add batch images from image6_14
        if image6_14 is not None:
            # image6_14 is [B, H, W, C]
            batch_size = image6_14.shape[0]
            for i in range(batch_size):
                input_tensors.append(image6_14[i:i+1]) # Keep dims as [1, H, W, C]

        # Enforce limits
        total_inputs = len(input_tensors)
        if total_inputs == 0:
             raise ValueError("At least one image input is required.")
        
        # Enforce strict 14 limit as requested
        if total_inputs > 14:
            print(f"NanoGemini Warning: {total_inputs} images provided. Truncating to 14 (Gemini 3 Pro limit).")
            input_tensors = input_tensors[:14]

        # Best-effort guidance from current model capabilities
        if len(input_tensors) > caps["max_images"]:
            print(
                f"NanoGemini Warning: {len(input_tensors)} images provided for '{model}'. "
                f"Recommended max is {caps['max_images']} for best quality."
            )

        input_parts = []

        # Process input images
        estimated_payload_bytes = len(prompt.encode("utf-8"))
        for i, img_tensor in enumerate(input_tensors):
            pil_img = tensor2pil(img_tensor)
            if pil_img:
                buffer = BytesIO()
                pil_img.save(buffer, format="PNG") # PNG is robust
                raw_bytes = buffer.getvalue()
                if use_file_api:
                    try:
                        file_uri = _upload_file_via_api(
                            raw_bytes,
                            "image/png",
                            f"nanogemini_input_{i+1}.png",
                            gemini_api_key,
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to upload input image {i+1} via File API: {e}") from e

                    input_parts.append({
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": file_uri
                        }
                    })
                else:
                    b64_data = base64.b64encode(raw_bytes).decode("utf-8")
                    estimated_payload_bytes += len(b64_data)
                    
                    # FIX: Use camelCase keys for REST API (inlineData, mimeType)
                    input_parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64_data
                        }
                    })

        # For multimodal understanding quality, keep text instruction after image parts.
        input_parts.append({"text": prompt})

        if not use_file_api and estimated_payload_bytes > SAFE_INLINE_BUDGET_BYTES:
            raise ValueError(
                f"Estimated inline payload is too large ({estimated_payload_bytes / (1024*1024):.2f} MB). "
                f"Reduce image count/resolution to stay under ~{MAX_INLINE_REQUEST_BYTES / (1024*1024):.0f} MB request limits, "
                f"or enable the File API toggle."
            )

        # 2. Configure API Payload
        safety_map = {
            "block_none": "BLOCK_NONE",
            "block_few": "BLOCK_ONLY_HIGH",
            "block_some": "BLOCK_MEDIUM_AND_ABOVE",
            "block_most": "BLOCK_LOW_AND_ABOVE"
        }
        
        safety_categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
        safety_settings = [
            {"category": cat, "threshold": safety_map[safety_filter]} for cat in safety_categories
        ]

        image_config = {"aspectRatio": aspect_ratio}
        # 2.5 works best with ratio-only config; 3.x supports imageSize control.
        if model != "gemini-2.5-flash-image":
            image_config["imageSize"] = api_resolution

        normalized_seed = _normalize_seed_for_api(seed)

        generation_config = {
            "responseModalities": ["IMAGE", "TEXT"],
            "candidateCount": 1,
            "imageConfig": image_config,
            "seed": normalized_seed,
        }

        # For 3.1 Flash Image, map toggle to thinking level.
        if model == "gemini-3.1-flash-image-preview":
            generation_config["thinkingConfig"] = {
                "thinkingLevel": "HIGH" if enable_thinking else "MINIMAL",
                "includeThoughts": False,
            }

        payload = {
            # Explicitly wrapping in role="user" is safer for newer Pro models
            "contents": [{
                "role": "user",
                "parts": input_parts
            }],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        if use_google_search:
            payload["tools"] = [{"google_search": {}}]

        # --- DEBUG OUTPUT ---
        if debug_payload:
            # Create a safe copy for printing (don't print 20MB of base64 text)
            debug_print = copy.deepcopy(payload)
            total_size_estimate = 0
            
            # Truncate base64 strings in the copy
            for part in debug_print["contents"][0]["parts"]:
                if "inlineData" in part:
                    data_len = len(part["inlineData"]["data"])
                    total_size_estimate += data_len
                    part["inlineData"]["data"] = f"<Base64 Image Data: {data_len} chars>"
                elif "fileData" in part:
                    part["fileData"]["fileUri"] = "<Uploaded File URI>"
            
            print("--- NANO GEMINI DEBUG PAYLOAD START ---")
            print(json.dumps(debug_print, indent=2))
            print(f"Approximate Payload Size (Images Only): {total_size_estimate / (1024*1024):.2f} MB")
            if not use_file_api and estimated_payload_bytes > SAFE_INLINE_BUDGET_BYTES:
                print("WARNING: Payload exceeds safe inline budget and may fail near 20MB request limits.")
            print("--- NANO GEMINI DEBUG PAYLOAD END ---")

        # 3. Define the Request Function
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}

        def send_request(idx):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                # Keep the original request number because num_images uses parallel requests.
                data["_request_index"] = idx + 1

                _log_gemini_usage(
                    model=model,
                    request_idx=idx + 1,
                    usage=data.get("usageMetadata"),
                    input_image_count=len(input_tensors),
                )

                return data
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                detail = e.response.text if e.response is not None else str(e)
                print(f"Gemini API Request {idx+1} Failed (HTTP {status}): {detail}")
                return {"_error": f"HTTP {status}: {detail}"}
            except requests.exceptions.RequestException as e:
                print(f"Gemini API Request {idx+1} Failed (Network): {e}")
                return {"_error": f"Network error: {e}"}
            except Exception as e:
                print(f"Gemini API Request {idx+1} Failed: {e}")
                return None

        # 4. Execute Parallel Requests (if num_images > 1)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_images)]
            
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        if not results:
             raise ValueError("All API requests failed. Check your API key and console logs.")

        # 5. Process Responses and Convert to Tensors
        output_tensors = []
        collected_errors = []

        for i, res in enumerate(results):
            try:
                if "_error" in res:
                    collected_errors.append(f"Req {i+1}: {res['_error'][:300]}")
                    continue

                candidates = res.get("candidates", [])
                
                if "promptFeedback" in res:
                    pf = res["promptFeedback"]
                    if pf.get("blockReason"):
                         print(f"Request {i+1} Blocked: {pf['blockReason']}")

                if not candidates:
                    collected_errors.append(f"Req {i+1}: No candidates returned.")
                    continue

                for candidate in candidates:
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    image_found_in_candidate = False
                    text_response = ""

                    for part in parts:
                        if "text" in part:
                            text_response += part["text"]
                        
                        # FIX: Use camelCase 'inlineData' for REST API response parsing
                        inline_data = part.get("inlineData", {})
                        if inline_data:
                            b64_img = inline_data.get("data")
                            if b64_img:
                                try:
                                    img_data = base64.b64decode(b64_img)
                                    pil_out = Image.open(BytesIO(img_data))
                                    # Normalize output format so downstream receives deterministic encoding behavior.
                                    normalized = _encode_output_preview(pil_out, output_mime_type)
                                    tensor_out = pil2tensor(normalized)
                                    output_tensors.append(tensor_out)
                                    image_found_in_candidate = True
                                except Exception as e:
                                    print(f"Error decoding image in Req {i+1}: {e}")

                    if not image_found_in_candidate:
                        msg = f"Req {i+1}: FinishReason={finish_reason}"
                        if text_response:
                            msg += f", Text='{text_response[:200]}...'"
                        collected_errors.append(msg)
                        
                        print(f"--- Full JSON Response for Req {i+1} (No Image Found) ---")
                        print(json.dumps(res, indent=2))
                        print("---------------------------------------------------------")

            except Exception as e:
                print(f"Error parsing response {i+1}: {e}")
                continue

        if not output_tensors:
            error_summary = "; ".join(collected_errors)
            raise ValueError(f"No images found in API responses. Details: {error_summary}")

        # 6. Stack and Return
        result_batch = torch.cat(output_tensors, dim=0)
        return (result_batch,)
