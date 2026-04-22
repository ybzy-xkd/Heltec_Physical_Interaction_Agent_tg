#!/usr/bin/env python3
"""
OpenClaw Image Workaround Script
Directly calls Ollama API to describe an image, bypassing OpenClaw image tool issues.

Usage:
    python3 describe_image.py <image_path> [model_name] [prompt]
    
Examples:
    python3 describe_image.py /path/to/image.jpg
    python3 describe_image.py /path/to/image.jpg blaifa/InternVL3_5:8B
    python3 describe_image.py /path/to/image.jpg blaifa/InternVL3_5:8B "Read only the text in this image"
"""

import sys
import requests
import base64
import json
import os

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "").strip()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_SEC = int(os.environ.get("OPENCLAW_LOCAL_VISION_TIMEOUT_SEC", "60").strip() or "60")


def describe_image(image_path: str, model: str = DEFAULT_MODEL, prompt: str = "Please describe this image in detail.") -> str:
    """Describe an image using Ollama API."""
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image and encode as base64
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt, "images": [img_data]}
        ],
        "stream": False
    }
    
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SEC)
    resp.raise_for_status()
    
    result = resp.json()
    return result['message']['content']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    image_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Please describe this image in detail."

    if not model:
        print("❌ Please set OLLAMA_MODEL to a model name available in your Ollama instance.")
        print("Example:")
        print("  OLLAMA_MODEL=xxx/yyy:zz")
        sys.exit(1)
    
    print(f"📷 Model: {model}")
    print(f"🖼️  Image: {image_path}")
    print(f"📝 Prompt: {prompt}")
    print("-" * 40)
    
    try:
        description = describe_image(image_path, model, prompt)
        print(description)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unknown error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
