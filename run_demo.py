"""run_demo.py

簡易範例：讀取 enhanced_prompt.txt 作為 system prompt，然後把使用者輸入送給模型。
請先把 OPENAI_API_KEY 設為環境變數。
"""
import os
import json
import argparse

try:
    from openai import OpenAI
except Exception:
    # 如果使用的 openai 版本不同，請安裝 openai >= 1.0.0
    raise


def load_system_prompt(path="enhanced_prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Run a demo of 員瑛式思考生成器")
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o"), help="Model name to use")
    parser.add_argument("--input", help="Event text to transform (if omitted, uses a sample)")
    parser.add_argument("--prompt", default="enhanced_prompt.txt", help="Path to the system prompt file")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[WARN] 未偵測到 OPENAI_API_KEY，將使用 Mock 模式輸出範例。若要使用真實模型，請設定環境變數。\n")

    system_prompt = load_system_prompt(args.prompt)
    user_input = args.input or "我今天忘了帶鑰匙，結果差點遲到"

    print("--- Request ---")
    print("Model:", args.model)
    print("User input:", user_input)

    if not api_key:
        # Produce a mock JSON that matches the enhanced_prompt schema
        mock = {
            "style": "貼文",
            "tone": "幽默",
            "content": f"[MOCK] {user_input} -> 這是一個模擬的員瑛式正向貼文。完全是 Lucky Vicky 呀!",
            "highlights": ["模擬摘要", "模擬學習點"],
        }
        text = json.dumps(mock, ensure_ascii=False)
    else:
        client = OpenAI()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        # Create chat completion
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=500,
        )

        # The new OpenAI SDK returns choices[].message.content
        try:
            text = resp.choices[0].message.content
        except Exception:
            # Fallback: try to stringify
            text = str(resp)

    print("\n--- Raw model output ---")
    print(text)

    # Try to parse JSON if model returned JSON
    try:
        parsed = json.loads(text)
        print("\n--- Parsed JSON ---")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception:
        print("\n(模型回傳非 JSON 或解析失敗。可嘗試指定 output 格式或調整 prompt。)")


if __name__ == "__main__":
    main()
