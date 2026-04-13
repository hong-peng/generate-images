import json
import urllib.request
import urllib.parse
import urllib.error
import time
import random
import base64
import os
from flask import Flask, request, jsonify, g

app = Flask(__name__)

COMFYUI_SERVER = "127.0.0.1:8188"
DEFAULT_CHECKPOINT = "realisticVisionV60B1_v51HyperVAE.safetensors"


@app.before_request
def log_request():
    g.start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.path}")


@app.after_request
def log_response(response):
    elapsed = time.time() - g.start_time
    status = "成功" if response.status_code < 400 else "失败"
    print(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.path} -> {response.status_code} {status} ({elapsed:.1f}s)")
    return response


def build_workflow(
    positive_prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    checkpoint: str = DEFAULT_CHECKPOINT,
):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "output", "images": ["6", 0]},
        },
    }


def build_img2img_workflow(
    positive_prompt: str,
    image_name: str,
    negative_prompt: str = "",
    steps: int = 20,
    cfg: float = 7.0,
    denoise: float = 0.75,
    seed: int = -1,
    checkpoint: str = DEFAULT_CHECKPOINT,
):
    """img2img workflow：以参考图为基础重新生成"""
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        # 加载上传的图片
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name, "upload": "image"},
        },
        # 将图片编码为 latent
        "5": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["4", 0], "vae": ["1", 2]},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": denoise,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "img2img", "images": ["7", 0]},
        },
    }


def upload_image(image_data: bytes, filename: str) -> str:
    """上传图片到 ComfyUI，返回文件名"""
    boundary = "----FormBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    result = json.loads(urllib.request.urlopen(req).read())
    return result["name"]


def queue_prompt(workflow: dict) -> str:
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode("utf-8"))


def cancel_prompt(prompt_id: str):
    """从队列中取消指定任务"""
    data = json.dumps({"delete": [prompt_id]}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/queue",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req)
    except Exception:
        pass


def wait_for_result(prompt_id: str, timeout: int = 300) -> list[dict]:
    """等待生成完成，返回图片信息列表"""
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(
            f"http://{COMFYUI_SERVER}/history/{prompt_id}"
        ) as r:
            history = json.loads(r.read())
        if prompt_id in history:
            images = []
            for node_id, output in history[prompt_id]["outputs"].items():
                for img in output.get("images", []):
                    images.append(img)
            return images
        time.sleep(1)
    raise TimeoutError("生成超时")


def download_image_base64(filename: str, subfolder: str = "") -> str:
    """下载图片并转为 base64"""
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": "output"}
    )
    with urllib.request.urlopen(
        f"http://{COMFYUI_SERVER}/view?{params}"
    ) as r:
        return base64.b64encode(r.read()).decode("utf-8")


# ============ API 接口 ============

@app.route("/generate", methods=["POST"])
def generate():
    """
    生成图片接口

    请求体 (JSON):
    {
        "prompt": "a beautiful woman",          // 必填，正向提示词
        "negative_prompt": "ugly, blurry",      // 选填，负向提示词
        "width": 512,                           // 选填，默认 512
        "height": 512,                          // 选填，默认 512
        "steps": 20,                            // 选填，默认 20
        "cfg": 7.0,                             // 选填，默认 7.0
        "seed": -1,                             // 选填，-1 随机
        "checkpoint": "model.safetensors"       // 选填，默认 realisticVision
    }

    返回:
    {
        "success": true,
        "prompt_id": "xxx",
        "images": [
            {
                "filename": "output_00001_.png",
                "data": "base64字符串"           // 图片 base64
            }
        ]
    }
    """
    body = request.get_json()
    if not body or "prompt" not in body:
        return jsonify({"success": False, "error": "缺少 prompt 参数"}), 400

    try:
        workflow = build_workflow(
            positive_prompt=body["prompt"],
            negative_prompt=body.get("negative_prompt", ""),
            width=body.get("width", 512),
            height=body.get("height", 512),
            steps=body.get("steps", 20),
            cfg=body.get("cfg", 7.0),
            seed=body.get("seed", -1),
            checkpoint=body.get("checkpoint", DEFAULT_CHECKPOINT),
        )

        prompt_id = queue_prompt(workflow)
        try:
            images_info = wait_for_result(prompt_id)
        except (TimeoutError, Exception) as e:
            cancel_prompt(prompt_id)
            raise

        images = []
        for img in images_info:
            b64 = download_image_base64(img["filename"], img.get("subfolder", ""))
            images.append({"filename": img["filename"], "data": b64})

        return jsonify({"success": True, "prompt_id": prompt_id, "images": images})

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504


@app.route("/generate/img2img", methods=["POST"])
def generate_img2img():
    """
    图生图接口

    请求体 (multipart/form-data):
        image           必填，参考图片文件
        prompt          必填，正向提示词
        negative_prompt 选填，负向提示词
        steps           选填，默认 20
        cfg             选填，默认 7.0
        denoise         选填，重绘幅度 0.0~1.0，默认 0.75（越大变化越多）
        seed            选填，-1 随机
        checkpoint      选填，默认 realisticVision

    返回:
    {
        "success": true,
        "prompt_id": "xxx",
        "images": [{"filename": "img2img_00001_.png", "data": "base64字符串"}]
    }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "缺少 image 文件"}), 400
    if not request.form.get("prompt"):
        return jsonify({"success": False, "error": "缺少 prompt 参数"}), 400

    try:
        file = request.files["image"]
        image_data = file.read()
        filename = file.filename or "input.png"

        uploaded_name = upload_image(image_data, filename)

        workflow = build_img2img_workflow(
            positive_prompt=request.form["prompt"],
            image_name=uploaded_name,
            negative_prompt=request.form.get("negative_prompt", ""),
            steps=int(request.form.get("steps", 20)),
            cfg=float(request.form.get("cfg", 7.0)),
            denoise=float(request.form.get("denoise", 0.75)),
            seed=int(request.form.get("seed", -1)),
            checkpoint=request.form.get("checkpoint", DEFAULT_CHECKPOINT),
        )

        prompt_id = queue_prompt(workflow)
        try:
            images_info = wait_for_result(prompt_id)
        except Exception as e:
            cancel_prompt(prompt_id)
            raise

        images = []
        for img in images_info:
            b64 = download_image_base64(img["filename"], img.get("subfolder", ""))
            images.append({"filename": img["filename"], "data": b64})

        return jsonify({"success": True, "prompt_id": prompt_id, "images": images})

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504


@app.route("/models", methods=["GET"])
def list_models():
    """列出可用的模型"""
    with urllib.request.urlopen(
        f"http://{COMFYUI_SERVER}/object_info/CheckpointLoaderSimple"
    ) as r:
        info = json.loads(r.read())
    models = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    return jsonify({"success": True, "models": models})


@app.route("/health", methods=["GET"])
def health():
    """检查服务状态"""
    try:
        urllib.request.urlopen(f"http://{COMFYUI_SERVER}/system_stats")
        return jsonify({"success": True, "comfyui": "running"})
    except Exception:
        return jsonify({"success": False, "comfyui": "not running"}), 503


if __name__ == "__main__":
    print("API 服务启动: http://127.0.0.1:5000")
    print("接口列表:")
    print("  POST /generate          - 文生图")
    print("  POST /generate/img2img  - 图生图")
    print("  GET  /models            - 查看可用模型")
    print("  GET  /health            - 检查状态")
    app.run(host="0.0.0.0", port=5000, debug=False)
