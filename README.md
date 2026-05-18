# ComfyUI Nano Banana Edit Gemini

ComfyUI custom node for Gemini native image generation/editing with multimodal image input support.

## Supported Models

- `gemini-2.5-flash-image` (Nano Banana)
- `gemini-3-pro-image-preview` (Nano Banana Pro)
- `gemini-3.1-flash-image-preview` (Nano Banana 2)

## Main Features

- Multi-model image editing with prompt + up to 14 input reference images.
- Parallel request execution to generate multiple variations quickly.
- Model-aware validation for ratios, resolutions, and feature toggles.
- Google Search toggle for supported Gemini 3 image models.
- Thinking toggle for Gemini 3.1 Flash Image Preview.
- File API toggle for large or reusable input images.
- Output format control: `png`, `jpg`, `webp`.
- Debug payload mode with safe base64 truncation and request-size diagnostics.
- API key from node input or `GEMINI_API_KEY` environment variable.

## Node Inputs

### Required

- `prompt` (STRING, multiline)
- `model`:
  - `gemini-2.5-flash-image`
  - `gemini-3-pro-image-preview`
  - `gemini-3.1-flash-image-preview`
- `gemini_api_key` (STRING, optional if `GEMINI_API_KEY` env var exists)
- `seed` (INT)
- `aspect_ratio`:
  - Common: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`
  - Gemini 3.1 additions: `1:4`, `4:1`, `1:8`, `8:1`
- `resolution`: `0.5K`, `1K`, `2K`, `4K`
  - `0.5K` maps to Gemini API value `512` and is valid only for `gemini-3.1-flash-image-preview`
  - `gemini-2.5-flash-image` is constrained to `1K` in this node
- `output_format`: `png`, `jpg`, `webp`
- `num_images` (1-4, parallel requests)
- `use_file_api` (BOOLEAN)
- `use_google_search` (BOOLEAN)
- `enable_thinking` (BOOLEAN)
  - Applies to `gemini-3.1-flash-image-preview` only
- `safety_filter`: `block_none`, `block_few`, `block_some`, `block_most`
- `debug_payload` (BOOLEAN)

### Optional image inputs

- `image1` ... `image5` (single image sockets)
- `image6_14` (batched IMAGE socket)

## Model-Specific Behavior

- `gemini-3.1-flash-image-preview`
  - Supports `0.5K`, `1K`, `2K`, `4K`
  - Supports expanded aspect ratios (`1:4`, `4:1`, `1:8`, `8:1`)
  - Thinking toggle maps to `thinkingConfig.thinkingLevel` (`HIGH` vs `MINIMAL`)
  - Supports Google Search tool toggle

- `gemini-3-pro-image-preview`
  - Supports `1K`, `2K`, `4K`
  - Supports base aspect-ratio set
  - Supports Google Search tool toggle

- `gemini-2.5-flash-image`
  - Uses ratio-only image configuration in this node
  - Resolution constrained to `1K` in this implementation
  - Google Search and Thinking toggle are not enabled for this model in this node

## Multi-Image Reference Limits (Latest Guidance)

This node accepts up to 14 input references and truncates extra inputs with a warning.

Latest model guidance for up to 14 references:

- `gemini-3.1-flash-image-preview`
  - Up to 10 object references with high-fidelity detail
  - Up to 4 character-consistency references
- `gemini-3-pro-image-preview`
  - Up to 6 object references with high-fidelity detail
  - Up to 5 character-consistency references

The node cannot automatically classify "object" vs "character" images, so it enforces only the global 14-image cap and prints warnings when count exceeds per-model recommended ranges.

## Image Understanding and Quality Notes

To improve quality and instruction following:

- Keep source images clear, correctly oriented, and not blurry.
- Prefer concise but explicit prompts with concrete edits.
- For multimodal prompts, this node sends image parts before text instruction for better image-understanding behavior.
- Keep inline payloads below API size limits.

## Request Size and Error Handling

- API inline payloads are validated with a conservative preflight budget.
- If estimated inline size is too large and File API is off, the node fails early with a clear message suggesting enabling File API.
- When File API is on, images are uploaded first and the node sends `fileData` references instead of inline base64 image parts.
- File API upload start/finalize failures are surfaced with explicit HTTP details.
- HTTP/network errors include per-request diagnostics.
- Candidate parsing failures include finish reason + snippet of text response when no image is returned.
- Safety blocks from `promptFeedback` are surfaced in logs.
- Gemini usage logs are written to `output/gemini_usage.jsonl` with token counts, estimated cost, `user_id`, and `username` when ComfyUI-Sentinel is installed and has an active user.

## Output Format Support

This node supports output normalization to:

- `image/png`
- `image/jpeg`
- `image/webp`

`HEIC` and `HEIF` output are intentionally skipped in this version.

## Security Best Practices

Use environment variables for API keys whenever possible.

> Do not publish workflows that include your API key in node text fields.

### Set `GEMINI_API_KEY`

#### Windows

1. Open **Edit the system environment variables**.
2. Open **Environment Variables** and add:
   - Name: `GEMINI_API_KEY`
   - Value: `your_actual_api_key_here`
3. Restart ComfyUI after changes.

#### Linux/macOS

Add to shell profile:

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

## References

- Gemini image generation docs: <https://ai.google.dev/gemini-api/docs/image-generation>
- 14 reference images details: <https://ai.google.dev/gemini-api/docs/image-generation#use-14-images>
- Gemini image understanding docs: <https://ai.google.dev/gemini-api/docs/image-understanding>
