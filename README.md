# ComfyUI Image Generation API

A lightweight HTTP API server wrapping [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for local AI image generation. Supports text-to-image, image-to-image, and animated GIF generation via AnimateDiff. Optimized for Apple Silicon (M-series) Macs.

## Features

- **Text-to-image** — generate images from a text prompt
- **Image-to-image** — redraw a reference image guided by a prompt
- **AnimateDiff** — generate animated GIFs from a text prompt
- **REST API** — Flask-based HTTP server, easy to integrate into any project
- **Queue safety** — failed tasks are automatically removed from the ComfyUI queue so they never block subsequent requests

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 or later)
- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally on `127.0.0.1:8188`
- A Stable Diffusion 1.5 checkpoint, e.g. [Realistic Vision V6](https://civitai.com/models/4201)
- *(Optional)* AnimateDiff motion module `mm_sd_v15_v2.ckpt` for GIF generation

## Installation

### 1. Clone and set up ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### 2. Add your model

Download a Stable Diffusion 1.5 checkpoint and place it in:

```
ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors
```

*(Optional)* For AnimateDiff, place the motion module in:

```
ComfyUI/models/animatediff_models/mm_sd_v15_v2.ckpt
```

### 3. Start ComfyUI

```bash
cd ComfyUI
python main.py --force-fp16
```

ComfyUI will be available at `http://127.0.0.1:8188`.

### 4. Install API server dependencies

```bash
pip install flask
```

### 5. Start the API server

```bash
python api_server.py
```

The API server will be available at `http://127.0.0.1:5000`.

---

## API Reference

### `POST /generate` — Text-to-image

**Request** (`application/json`):

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | Yes | — | Positive prompt |
| `negative_prompt` | string | No | `""` | Negative prompt |
| `width` | int | No | `512` | Image width |
| `height` | int | No | `512` | Image height |
| `steps` | int | No | `20` | Sampling steps |
| `cfg` | float | No | `7.0` | CFG scale |
| `seed` | int | No | `-1` | Seed (-1 = random) |
| `checkpoint` | string | No | `realisticVisionV60B1_v51HyperVAE.safetensors` | Model filename |

**Example:**

```bash
curl -X POST "http://127.0.0.1:5000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful woman, sunset, photorealistic, 8k",
    "negative_prompt": "ugly, blurry, low quality",
    "width": 512,
    "height": 512,
    "steps": 20
  }'
```

---

### `POST /generate/img2img` — Image-to-image

**Request** (`multipart/form-data`):

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | file | Yes | — | Reference image |
| `prompt` | string | Yes | — | Positive prompt |
| `negative_prompt` | string | No | `""` | Negative prompt |
| `steps` | int | No | `20` | Sampling steps |
| `cfg` | float | No | `7.0` | CFG scale |
| `denoise` | float | No | `0.75` | Redraw strength (0.0–1.0) |
| `seed` | int | No | `-1` | Seed (-1 = random) |
| `checkpoint` | string | No | `realisticVisionV60B1_v51HyperVAE.safetensors` | Model filename |

> **`denoise` guide:** `0.3` = subtle changes, `0.75` = balanced, `1.0` = full redraw (same as text-to-image)

**Example:**

```bash
curl -X POST "http://127.0.0.1:5000/generate/img2img" \
  -F "image=@/path/to/your_image.png" \
  -F "prompt=a beautiful woman, sunset, photorealistic, 8k" \
  -F "negative_prompt=ugly, blurry, low quality" \
  -F "denoise=0.75"
```

---

### `GET /models` — List available models

```bash
curl "http://127.0.0.1:5000/models"
```

---

### `GET /health` — Check service status

```bash
curl "http://127.0.0.1:5000/health"
```

---

### Response format

All endpoints return the same structure:

```json
{
  "success": true,
  "prompt_id": "eda9b6a9-xxxx",
  "images": [
    {
      "filename": "output_00001_.png",
      "data": "<base64-encoded image>"
    }
  ]
}
```

On error:

```json
{
  "success": false,
  "error": "error message"
}
```

---

## AnimateDiff (GIF generation)

Run the standalone script directly (not part of the API server):

```bash
python generate_video.py
```

Output is saved as a `.gif` file in the current directory. Edit the script to change the prompt, resolution, frame count, and FPS.

---

## Queue Management

If the ComfyUI queue gets stuck, use these commands to clear it:

```bash
# Clear all pending tasks
curl -X POST "http://127.0.0.1:8188/queue" \
  -H "Content-Type: application/json" \
  -d '{"clear": true}'

# Interrupt the currently running task
curl -X POST "http://127.0.0.1:8188/interrupt" \
  -H "Content-Type: application/json"
```

## Project Structure

```
.
├── api_server.py              # Flask HTTP API server
├── generate.py                # Standalone text-to-image script
├── generate_video.py          # Standalone AnimateDiff GIF script
├── ComfyUI/                   # Not included — clone separately (see Installation)
└── stable-diffusion-webui/    # Not included — clone separately (see Installation)
```

> `ComfyUI/` and `stable-diffusion-webui/` are excluded from this repository via `.gitignore`.
> Clone them manually following the Installation steps above.
