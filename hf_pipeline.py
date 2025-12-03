"""hf_pipeline.py

示範流程：
- 使用 Hugging Face Inference API 的 Whisper 模型做 ASR（audio -> text）
- 再把轉寫結果送到 HF Router chat completions（或其他 HF 聊天模型）

安全性：不會把 token 寫在程式中，會從環境變數 HF_API_TOKEN 讀取。

使用方式（PowerShell 範例）:
  $env:HF_API_TOKEN = '你的_new_token'
  python hf_pipeline.py --audio api.wav --hf_model openai/whisper-small --chat_model openai/gpt-oss-20b:together

"""
import os
import sys
import argparse
import requests
from typing import Any, Dict


def transcribe_audio_requests(audio_path: str, hf_model: str, hf_token: str) -> str:
    api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    with open(audio_path, "rb") as f:
        data = f.read()
    resp = requests.post(api_url, headers=headers, data=data)
    resp.raise_for_status()
    result = resp.json()
    # Defensive parsing
    if isinstance(result, dict):
        # common keys
        for k in ("text", "transcription", "sentence"):
            if k in result and result[k]:
                return result[k]
        # sometimes model returns {'chunks': [...]} or other structure
        if "chunks" in result and isinstance(result["chunks"], list):
            texts = [c.get("text") or c.get("transcription") for c in result["chunks"] if isinstance(c, dict)]
            return " ".join([t for t in texts if t])
        return str(result)
    return str(result)


def hf_chat_router(messages: Any, chat_model: str, hf_token: str) -> Dict:
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"messages": messages, "model": chat_model}
    resp = requests.post(api_url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="HF pipeline: audio -> whisper -> chat via HF Router")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3)")
    parser.add_argument("--hf_model", default="openai/whisper-small", help="HF ASR model repo id")
    parser.add_argument("--chat_model", default="openai/gpt-oss-20b:together", help="HF chat model id for Router")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        print("ERROR: HF_API_TOKEN not set. Please set it in your environment before running.")
        print("PowerShell example: $env:HF_API_TOKEN = 'hf_xxx'")
        sys.exit(1)

    # 1) ASR
    print(f"Transcribing {args.audio} with model {args.hf_model}...")
    try:
        transcript = transcribe_audio_requests(args.audio, args.hf_model, hf_token)
    except Exception as e:
        print("ASR request failed:", e)
        sys.exit(1)

    print("--- Transcript ---")
    print(transcript)

    # 2) Chat - send transcript as user message
    messages = [{"role": "user", "content": transcript}]
    print(f"Sending transcript to chat model {args.chat_model}...")
    try:
        resp = hf_chat_router(messages, args.chat_model, hf_token)
    except Exception as e:
        print("Chat request failed:", e)
        sys.exit(1)

    # Defensive parsing of router response
    try:
        choice = resp.get("choices")[0]
        # choice may contain 'message' or 'text'
        msg = None
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("text") or choice
        else:
            msg = choice
        print("--- Chat response ---")
        print(msg)
    except Exception:
        print("Unexpected chat response format:")
        print(resp)


if __name__ == "__main__":
    main()
