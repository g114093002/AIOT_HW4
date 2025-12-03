"""hf_examples.py

示範如何安全地使用 Hugging Face Inference API：
- Whisper ASR (audio -> text)
- Router chat completions (chat -> model)

注意：不要把 token 硬編在程式裡，請以環境變數 HF_API_TOKEN 設定。
"""
import os
import requests
from typing import Any, Dict


def transcribe_audio_requests(audio_path: str, hf_model: str = "openai/whisper-small") -> str:
    """Use HF Inference direct model endpoint via requests to transcribe an audio file.

    This function reads HF_API_TOKEN from environment. It returns the transcript string
    or raises RuntimeError on failure.
    """
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN environment variable not set")

    api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    with open(audio_path, "rb") as f:
        data = f.read()

    resp = requests.post(api_url, headers=headers, data=data)
    resp.raise_for_status()
    result = resp.json()
    # HF may return plain text or a JSON with 'text' key depending on wrapper
    if isinstance(result, dict):
        return result.get("text") or result.get("transcription") or str(result)
    return str(result)


def hf_chat_router(messages: Any, model: str = "openai/gpt-oss-20b:together") -> Dict:
    """Call HF Router chat completions endpoint.

    messages: a list-like of dicts with 'role' and 'content'.
    Returns the JSON response (dict) or raises on HTTP error.
    """
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN environment variable not set")

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"messages": messages, "model": model}

    resp = requests.post(api_url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Quick demo (requires HF_API_TOKEN in environment)
    try:
        print("Demo: HF Router chat (no token hardcoded)")
        resp = hf_chat_router([{"role": "user", "content": "What is the capital of France?"}])
        # defensive parsing
        choice = resp.get("choices") and resp["choices"][0]
        if choice and isinstance(choice, dict):
            msg = choice.get("message") or choice.get("text") or choice
            print("->", msg)
        else:
            print(resp)
    except Exception as e:
        print("HF demo error:", e)
