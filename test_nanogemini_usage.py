#!/usr/bin/env python3

import argparse
import base64
import copy
import json
import os
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


SAFETY_MAP = {
    "block_none": "BLOCK_NONE",
    "block_few": "BLOCK_ONLY_HIGH",
    "block_some": "BLOCK_MEDIUM_AND_ABOVE",
    "block_most": "BLOCK_LOW_AND_ABOVE",
}

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

PRICING_SOURCE_URLS = {
    "standard": "https://ai.google.dev/gemini-api/docs/pricing#standard_3",
    "batch": "https://ai.google.dev/gemini-api/docs/pricing#batch_3",
}
DEFAULT_COST_TIERS = ("standard",)

# USD per 1M tokens, based on the Gemini Developer API pricing page.
# Image model output can have separate TEXT and IMAGE token prices.
PRICING_USD_PER_1M = {
    "gemini-2.5-flash-image": {
        "standard": {"input": 0.30, "output_text": 2.50, "output_image": 30.00},
        "batch": {"input": 0.15, "output_text": 1.25, "output_image": 15.00},
        "flex": {"input": 0.15, "output_text": 1.25, "output_image": 15.00},
        "priority": {"input": 0.54, "output_text": 4.50, "output_image": 54.4186046512},
    },
    "gemini-3.1-flash-image-preview": {
        "standard": {"input": 0.50, "output_text": 3.00, "output_image": 60.00},
        "batch": {"input": 0.25, "output_text": 1.50, "output_image": 30.00},
    },
    "gemini-3-pro-image-preview": {
        "standard": {"input": 2.00, "output_text": 12.00, "output_image": 120.00},
        "batch": {"input": 1.00, "output_text": 6.00, "output_image": 60.00},
        "flex": {"input": 1.00, "output_text": 6.00, "output_image": 60.00},
        "priority": {"input": 3.60, "output_text": 21.60, "output_image": 216.00},
    },
}


def normalize_seed_for_api(seed):
    """Gemini generationConfig.seed expects int32."""
    max_i32 = 2147483647
    min_i32 = -2147483648

    seed = int(seed)
    if min_i32 <= seed <= max_i32:
        return seed

    return seed % (max_i32 + 1)


def image_file_to_inline_part(image_path):
    """Load image, convert to RGB PNG, then send as Gemini inlineData."""
    img = Image.open(image_path).convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    raw_bytes = buffer.getvalue()

    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(raw_bytes).decode("utf-8"),
        }
    }


def sanitize_response_for_debug(data):
    """Remove huge base64 strings before writing debug JSON."""
    clean = copy.deepcopy(data)

    for candidate in clean.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline and "data" in inline:
                inline["data"] = f"<base64 omitted, {len(inline['data'])} chars>"

    return clean


def tokens_by_modality(details):
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


def estimate_cost(usage, model, pricing_tier):
    pricing_by_tier = PRICING_USD_PER_1M.get(model)
    if not pricing_by_tier:
        return {
            "pricingAvailable": False,
            "pricingNote": f"No local pricing table for model '{model}'.",
        }

    rates = pricing_by_tier.get(pricing_tier)
    if not rates:
        supported = ", ".join(sorted(pricing_by_tier))
        return {
            "pricingAvailable": False,
            "pricingNote": (
                f"Pricing tier '{pricing_tier}' is not configured for model '{model}'. "
                f"Supported tiers: {supported}."
            ),
        }

    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount") or 0)

    prompt_by_modality = tokens_by_modality(usage.get("promptTokensDetails"))
    candidate_by_modality = tokens_by_modality(usage.get("candidatesTokensDetails"))

    if prompt_by_modality:
        input_tokens = sum(prompt_by_modality.values())
    else:
        input_tokens = prompt_tokens

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

    text_output_cost = (
        (output_text_tokens + unclassified_output_tokens)
        * rates["output_text"]
        / 1_000_000
    )
    image_output_cost = image_output_tokens * rates["output_image"] / 1_000_000
    input_cost = input_tokens * rates["input"] / 1_000_000
    total_cost = input_cost + text_output_cost + image_output_cost

    return {
        "pricingAvailable": True,
        "pricingTier": pricing_tier,
        "pricingSource": PRICING_SOURCE_URLS.get(
            pricing_tier,
            "https://ai.google.dev/gemini-api/docs/pricing",
        ),
        "ratesUsdPer1MTokens": rates,
        "inputTokensPriced": input_tokens,
        "outputTextTokensPriced": output_text_tokens,
        "outputImageTokensPriced": image_output_tokens,
        "unclassifiedOutputTokensPricedAsText": unclassified_output_tokens,
        "inputCostUsd": input_cost,
        "outputTextCostUsd": text_output_cost,
        "outputImageCostUsd": image_output_cost,
        "estimatedCostUsd": total_cost,
        "pricingNote": (
            "Estimate excludes separate tool charges such as Google Search grounding. "
            "Unclassified output tokens are priced as text output."
        ),
    }


def estimate_costs_by_tier(usage, model, pricing_tiers):
    return {
        tier: estimate_cost(usage, model, tier)
        for tier in pricing_tiers
    }


def print_cost_line(call_index, tier, cost):
    if cost and cost.get("pricingAvailable"):
        print(
            f"[call {call_index}] estimated {tier} cost: "
            f"${cost['estimatedCostUsd']:.8f} "
            f"(input=${cost['inputCostUsd']:.8f}, "
            f"text_out=${cost['outputTextCostUsd']:.8f}, "
            f"image_out=${cost['outputImageCostUsd']:.8f})",
            flush=True,
        )
    elif cost:
        print(f"[call {call_index}] {tier} cost unavailable: {cost['pricingNote']}", flush=True)


def log_usage(data, model, call_index, log_path, pricing_tiers):
    usage = data.get("usageMetadata")

    if not isinstance(usage, dict):
        print(f"[call {call_index}] usageMetadata missing", flush=True)
        return None

    costs_by_tier = estimate_costs_by_tier(usage, model, pricing_tiers)

    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "call": call_index,
        "model": model,
        "pricingTiers": list(pricing_tiers),
        "promptTokenCount": usage.get("promptTokenCount"),
        "candidatesTokenCount": usage.get("candidatesTokenCount"),
        "totalTokenCount": usage.get("totalTokenCount"),
        "thoughtsTokenCount": usage.get("thoughtsTokenCount"),
        "toolUsePromptTokenCount": usage.get("toolUsePromptTokenCount"),
        "cachedContentTokenCount": usage.get("cachedContentTokenCount"),
        "promptTokensDetails": usage.get("promptTokensDetails"),
        "candidatesTokensDetails": usage.get("candidatesTokensDetails"),
        "cacheTokensDetails": usage.get("cacheTokensDetails"),
        "toolUsePromptTokensDetails": usage.get("toolUsePromptTokensDetails"),
        "costs": costs_by_tier,
    }

    print(f"\n[call {call_index}] Gemini usage:")
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for tier, cost in costs_by_tier.items():
        print_cost_line(call_index, tier, cost)

    return costs_by_tier


def save_output_images(data, call_index, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    ext_for_mime = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }

    for cand_i, candidate in enumerate(data.get("candidates", []), start=1):
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            print(f"[call {call_index}] candidate {cand_i} finishReason={finish_reason}")

        parts = candidate.get("content", {}).get("parts", [])

        for part_i, part in enumerate(parts, start=1):
            text = part.get("text", "")
            if text.strip():
                print(f"[call {call_index}] text part: {text[:500]}")

            inline = part.get("inlineData")
            if not inline:
                continue

            b64_data = inline.get("data")
            if not b64_data:
                continue

            mime_type = inline.get("mimeType", "image/png")
            ext = ext_for_mime.get(mime_type, "bin")
            image_bytes = base64.b64decode(b64_data)
            path = out_dir / f"call_{call_index}_candidate_{cand_i}_part_{part_i}.{ext}"
            path.write_bytes(image_bytes)

            print(f"[call {call_index}] saved image: {path}")
            saved += 1

    return saved


def build_payload(args):
    api_resolution = "512" if args.resolution == "0.5K" else args.resolution

    input_parts = []
    if args.image:
        input_parts.append(image_file_to_inline_part(Path(args.image)))
    input_parts.append({"text": args.prompt})

    image_config = {
        "aspectRatio": args.aspect_ratio,
    }

    # Match the ComfyUI node: 2.5 Flash Image uses ratio-only config.
    if args.model != "gemini-2.5-flash-image":
        image_config["imageSize"] = api_resolution

    generation_config = {
        "responseModalities": ["IMAGE", "TEXT"],
        "candidateCount": 1,
        "imageConfig": image_config,
        "seed": normalize_seed_for_api(args.seed),
    }

    if args.model == "gemini-3.1-flash-image-preview":
        generation_config["thinkingConfig"] = {
            "thinkingLevel": "HIGH" if args.enable_thinking else "MINIMAL",
            "includeThoughts": False,
        }

    safety_settings = [
        {"category": category, "threshold": SAFETY_MAP[args.safety_filter]}
        for category in SAFETY_CATEGORIES
    ]

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": input_parts,
            }
        ],
        "generationConfig": generation_config,
        "safetySettings": safety_settings,
    }

    if args.use_google_search:
        payload["tools"] = [{"google_search": {}}]

    return payload


def call_gemini(args, call_index):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{args.model}:generateContent?key={api_key}"
    )

    payload = build_payload(args)
    headers = {"Content-Type": "application/json"}

    print(f"\n[call {call_index}] sending request to model={args.model}")
    response = requests.post(url, headers=headers, json=payload, timeout=args.timeout)

    if not response.ok:
        print(f"[call {call_index}] HTTP {response.status_code}")
        print(response.text[:4000])
        response.raise_for_status()

    data = response.json()

    if args.debug_response:
        debug_path = Path(args.out_dir) / f"call_{call_index}_response_sanitized.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps(sanitize_response_for_debug(data), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[call {call_index}] wrote sanitized response: {debug_path}")

    costs_by_tier = log_usage(
        data=data,
        model=args.model,
        call_index=call_index,
        log_path=Path(args.log),
        pricing_tiers=args.cost_tiers,
    )

    saved = save_output_images(
        data=data,
        call_index=call_index,
        out_dir=Path(args.out_dir),
    )

    if saved == 0:
        print(f"[call {call_index}] no output image found in response")

    return costs_by_tier


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Gemini image generation/edit usageMetadata test."
    )

    parser.add_argument(
        "--image",
        help="Optional input image path. Omit for prompt-to-image generation.",
    )
    parser.add_argument(
        "--prompt",
        default="Create a warm cinematic product photo of a ceramic coffee mug on a wooden table.",
        help="Generation or edit prompt.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help=(
            "Example: gemini-2.5-flash-image, "
            "gemini-3.1-flash-image-preview, gemini-3-pro-image-preview"
        ),
    )
    parser.add_argument("--num-calls", type=int, default=1)
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument(
        "--resolution",
        default="1K",
        choices=["0.5K", "512", "1K", "2K", "4K"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--safety-filter",
        default="block_none",
        choices=list(SAFETY_MAP.keys()),
    )
    parser.add_argument("--use-google-search", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log", default="gemini_usage_test.jsonl")
    parser.add_argument("--out-dir", default="gemini_test_outputs")
    parser.add_argument("--debug-response", action="store_true")
    parser.add_argument(
        "--cost-tiers",
        nargs="+",
        default=list(DEFAULT_COST_TIERS),
        choices=["standard", "batch", "flex", "priority"],
        help="Pricing tables to show. Defaults to standard.",
    )

    args = parser.parse_args()

    if args.num_calls < 1:
        raise ValueError("--num-calls must be >= 1")

    total_costs = {tier: 0.0 for tier in args.cost_tiers}
    priced_calls = {tier: 0 for tier in args.cost_tiers}

    for call_index in range(1, args.num_calls + 1):
        costs_by_tier = call_gemini(args, call_index)
        if not costs_by_tier:
            continue

        for tier, cost in costs_by_tier.items():
            if cost and cost.get("pricingAvailable"):
                total_costs[tier] += cost["estimatedCostUsd"]
                priced_calls[tier] += 1

    if any(priced_calls.values()):
        print("\n[total] estimated costs:")
        for tier in args.cost_tiers:
            if priced_calls[tier]:
                print(
                    f"[total] {tier}: ${total_costs[tier]:.8f} "
                    f"for {priced_calls[tier]} priced call(s)"
                )
        print("[total] pricing sources:")
        for tier in args.cost_tiers:
            source = PRICING_SOURCE_URLS.get(tier, "https://ai.google.dev/gemini-api/docs/pricing")
            print(f"[total] {tier}: {source}")


if __name__ == "__main__":
    main()
