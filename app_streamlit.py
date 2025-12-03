"""Streamlit app for 員瑛式思考生成器 — deployable with `streamlit run app_streamlit.py`"""
import os
import json

import streamlit as st
from openai import OpenAI
from openai.error import OpenAIError
from pathlib import Path

# Optional HF client
try:
    from huggingface_hub import InferenceApi
except Exception:
    InferenceApi = None

BASE_PROMPT_PATH = Path(__file__).parent / "enhanced_prompt.txt"

PERSONA_PREFIX = {
    "員瑛 (原版)": "請用員瑛式思考，保持正向、幽默，並以「完全是 Lucky Vicky 呀!」收尾。",
    "嚴肅": "請用沉穩且專業的語氣，給予實際建議，結尾保留正向鼓勵但不誇張。",
    "搞笑": "請用活潑搞笑的語氣，把小事誇張成趣事，以幽默收尾。",
}


def load_system_prompt(path: str = None) -> str:
    path = path or BASE_PROMPT_PATH
    return open(path, "r", encoding="utf-8").read()


def call_model(system_prompt: str, user_input: str, model: str, temperature: float, api_key: str | None = None):
    """Call the OpenAI Chat Completions API. Pass api_key explicitly to make authentication clear.

    Raises OpenAIError on API-level errors so callers can handle it and present user-friendly guidance.
    """
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=500,
        )
    except OpenAIError:
        # Re-raise so the UI layer can show friendly advice without leaking secrets
        raise
    except Exception as e:
        # Wrap other exceptions into a generic OpenAIError for unified handling
        raise OpenAIError(str(e))

    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


def call_hf_model(system_prompt: str, user_input: str, model: str, temperature: float, hf_token: str):
    """Call Hugging Face Inference API via huggingface_hub.InferenceApi.

    Returns generated text (string)."""
    if InferenceApi is None:
        raise RuntimeError("huggingface_hub not installed")
    prompt = system_prompt + "\n使用者：" + user_input
    api = InferenceApi(repo_id=model, token=hf_token)
    # parameters may vary by model; keep conservative defaults
    try:
        output = api(inputs=prompt, parameters={"max_new_tokens": 256, "temperature": temperature})
    except Exception as e:
        raise RuntimeError(str(e))
    # InferenceApi may return dict or str
    if isinstance(output, dict):
        # Common key 'generated_text' or list
        if "generated_text" in output:
            return output["generated_text"]
        # some models return [{'generated_text': '...'}]
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        return str(output)
    return str(output)


def mock_response(user_input: str, style: str, tone: str) -> str:
    """Generate a deterministic mock JSON response matching the prompt schema."""
    content = (
        f"{user_input} -> 這看起來其實是一件小小的幸運事！我把它想成一個正向的轉折，讓你更有能量面對下一步。完全是 Lucky Vicky 呀!"
    )
    mock = {
        "style": style,
        "tone": tone,
        "content": content,
        "highlights": ["把不便轉成趣事", "小事帶來正能量"],
    }
    return json.dumps(mock, ensure_ascii=False)


st.set_page_config(page_title="員瑛式思考生成器 (Streamlit)", layout="wide")

st.title("員瑛式思考生成器 — Streamlit 版")

with st.sidebar:
    st.header("設定")
    persona = st.selectbox("Persona", list(PERSONA_PREFIX.keys()), index=0)
    style = st.selectbox("Style", ["貼文", "短文", "鼓勵"]) 
    tone = st.selectbox("Tone", ["輕鬆", "熱情", "溫暖", "幽默"], index=3)
    model = st.text_input("Model", value=os.environ.get("MODEL", "gpt-4o"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    extra = st.text_area("額外指示（選填）", height=80)

st.markdown("請在下方輸入你想轉化的一段事件描述（1-3 句話）：")
user_input = st.text_area("事件描述", value="我今天忘了帶鑰匙，差點遲到。")

if st.button("產生"):
    base = load_system_prompt()
    persona_instr = PERSONA_PREFIX.get(persona, "")
    combined = f"{base}\n\n# Persona instruction:\n{persona_instr}\n請將輸出 style 設為：{style}，tone 設為：{tone}。\n{extra}"
    openai_key = os.environ.get("OPENAI_API_KEY")
    hf_token = os.environ.get("HF_API_TOKEN")
    hf_model = os.environ.get("HF_MODEL", os.environ.get("MODEL", "google/flan-t5-large"))

    # Priority: OpenAI -> HuggingFace -> Mock
    if openai_key:
        with st.spinner("正在使用 OpenAI 模型..."):
            try:
                out = call_model(combined, user_input, model=model, temperature=temperature, api_key=openai_key)
            except OpenAIError as e:
                st.error("OpenAI API 發生錯誤：無法完成請求。請檢查 Key 或帳戶狀態。")
                st.write(f"錯誤摘要：{str(e)}")
                st.stop()
            except Exception as e:
                st.error("發生非預期的錯誤：\n" + str(e))
                st.stop()
    elif hf_token:
        if InferenceApi is None:
            st.error("本環境未安裝 huggingface_hub；請安裝或使用 mock 模式。")
            out = mock_response(user_input, style, tone)
        else:
            with st.spinner("正在使用 Hugging Face 模型..."):
                try:
                    out = call_hf_model(combined, user_input, hf_model, temperature, hf_token)
                except Exception as e:
                    st.error("呼叫 Hugging Face Inference API 時發生錯誤：\n" + str(e))
                    st.stop()
    else:
        st.info("未偵測到 OPENAI_API_KEY 或 HF_API_TOKEN，啟用 Mock 模式（模擬回應）。若要使用真實模型請在 Streamlit Secrets 中設定相應的金鑰。")
        out = mock_response(user_input, style, tone)

        st.subheader("原始模型回傳")
        st.text_area("raw output", value=out, height=200)
        st.subheader("解析後 JSON (若可解析)")
        try:
            parsed = json.loads(out)
            st.json(parsed)
        except Exception:
            st.warning("模型回傳非 JSON 或解析失敗。請檢查 prompt 或嘗試使用預設參數。")
            st.write(out)

st.markdown("---")
st.markdown("部署範例：在專案目錄下執行 `streamlit run app_streamlit.py`（PowerShell）：")
st.code("$env:OPENAI_API_KEY='你的_API_KEY'; streamlit run app_streamlit.py", language="powershell")
