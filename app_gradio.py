"""Gradio app for 員瑛式思考生成器 — supports Persona switching and JSON visualization."""
import os
import json
from typing import Tuple

import gradio as gr

try:
    from huggingface_hub import InferenceApi
except Exception:
    InferenceApi = None
import os


def load_system_prompt(path: str = "enhanced_prompt.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PERSONA_PREFIX = {
    "員瑛 (原版)": "請用員瑛式思考，保持正向、幽默，並以「完全是 Lucky Vicky 呀!」收尾。",
    "嚴肅": "請用沉穩且專業的語氣，給予實際建議，結尾保留正向鼓勵但不誇張。",
    "搞笑": "請用活潑搞笑的語氣，把小事誇張成趣事，以幽默收尾。",
}


def call_model(system_prompt: str, user_input: str, model: str = "gpt-4o", temperature: float = 0.7):
    # legacy placeholder removed; use call_hf_model or mock via the UI logic
    raise RuntimeError("call_model is deprecated; use Hugging Face Inference (call_hf_model) or mock mode")


def format_for_display(raw: str) -> Tuple[str, dict]:
    """Return raw text and parsed JSON (or error)."""
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"error": "解析 JSON 失敗，原始回傳如下", "raw": raw}
    return raw, parsed


def build_system_prompt(base_prompt_path: str, persona: str, extra_instructions: str = "") -> str:
    base = load_system_prompt(base_prompt_path)
    persona_instr = PERSONA_PREFIX.get(persona, "")
    combined = f"{base}\n\n# Persona instruction:\n{persona_instr}\n{extra_instructions}"
    return combined


def call_hf_model(system_prompt: str, user_input: str, model: str, temperature: float, hf_token: str):
    if InferenceApi is None:
        raise RuntimeError("huggingface_hub not installed")
    prompt = system_prompt + "\n使用者：" + user_input
    api = InferenceApi(repo_id=model, token=hf_token)
    output = api(inputs=prompt, parameters={"max_new_tokens": 256, "temperature": temperature})
    if isinstance(output, dict):
        if "generated_text" in output:
            return output["generated_text"]
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        return str(output)
    return str(output)


def gradio_interface():
    with gr.Blocks(title="員瑛式思考生成器 — Gradio") as demo:
        gr.Markdown("# 員瑛式思考生成器 — Gradio 介面")
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(label="事件描述（1-3 句話）", lines=4, placeholder="例如：我今天忘了帶鑰匙，結果... ")
                persona = gr.Radio(list(PERSONA_PREFIX.keys()), value="員瑛 (原版)", label="Persona")
                style = gr.Dropdown(choices=["貼文", "短文", "鼓勵"], value="貼文", label="輸出風格 (style)")
                tone = gr.Dropdown(choices=["輕鬆", "熱情", "溫暖", "幽默"], value="幽默", label="語氣 (tone)")
                model = gr.Textbox(label="Model", value=os.environ.get("MODEL", "gpt-4o"))
                temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="Temperature")
                extra = gr.Textbox(label="額外指示 (選填)", lines=2, placeholder="例如：不要用 emoji")
                submit = gr.Button("產生")
            with gr.Column(scale=1):
                raw_out = gr.Textbox(label="原始模型回傳", lines=12)
                parsed_out = gr.JSON(label="解析後的 JSON")

        def on_submit(inp, persona_sel, style_sel, tone_sel, model_name, temp, extra_instr):
            # Build combined system prompt with light overrides
            extra = f"\n請將輸出 style 設為：{style_sel}，tone 設為：{tone_sel}。"
            if extra_instr:
                extra += f"\n額外指示：{extra_instr}"
            system = build_system_prompt("enhanced_prompt.txt", persona_sel, extra)
            user_text = inp or "我今天忘了帶鑰匙，結果差點遲到。"
            # Prefer Hugging Face if HF_API_TOKEN present; otherwise use mock mode
            hf_token = os.environ.get("HF_API_TOKEN")
            hf_model = os.environ.get("HF_MODEL", model_name)
            if hf_token:
                if InferenceApi is None:
                    raw = json.dumps({"error": "huggingface_hub not installed; cannot call HF"}, ensure_ascii=False)
                else:
                    try:
                        raw = call_hf_model(system, user_text, hf_model, temp, hf_token)
                    except Exception as e:
                        raw = json.dumps({"error": str(e)}, ensure_ascii=False)
            else:
                # mock placeholder
                raw = json.dumps({
                    "style": style_sel,
                    "tone": tone_sel,
                    "content": f"[MOCK] {user_text} -> 這是一個模擬回應。",
                    "highlights": ["模擬摘要"],
                }, ensure_ascii=False)
            raw2, parsed = format_for_display(raw)
            return raw2, parsed

        submit.click(on_submit, inputs=[user_input, persona, style, tone, model, temperature, extra], outputs=[raw_out, parsed_out])

    return demo


if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(share=False)
