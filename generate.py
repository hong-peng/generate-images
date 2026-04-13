import json
import urllib.request
import urllib.parse
import time
import random

SERVER = "127.0.0.1:8188"


def build_workflow(
    positive_prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    checkpoint: str = "realisticVisionV60B1_v51HyperVAE.safetensors",
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


def queue_prompt(workflow: dict) -> str:
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{SERVER}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"HTTP错误 {e.code}: {e.reason}")
        print("详细信息:", e.read().decode("utf-8"))
        raise


def get_image(prompt_id: str):
    print("生成中", end="", flush=True)
    while True:
        with urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            print(" 完成！")
            for node_id, output in history[prompt_id]["outputs"].items():
                for img in output.get("images", []):
                    params = urllib.parse.urlencode(
                        {
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": "output",
                        }
                    )
                    save_path = f"./{img['filename']}"
                    urllib.request.urlretrieve(
                        f"http://{SERVER}/view?{params}", save_path
                    )
                    print(f"已保存: {save_path}")
            return
        print(".", end="", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    workflow = build_workflow(
        positive_prompt="a cat sitting in table",
        negative_prompt="ugly, blurry, low quality, deformed, watermark",
    )
    prompt_id = queue_prompt(workflow)
    print(f"任务ID: {prompt_id}")
    get_image(prompt_id)
