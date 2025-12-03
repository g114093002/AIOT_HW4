"""Streamlit app for 員瑛式思考生成器 — deployable with `streamlit run app_streamlit.py`"""
import os
import json

import streamlit as st
from openai import OpenAI
from openai.error import OpenAIError

from pathlib import Path

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
    api_key = os.environ.get("OPENAI_API_KEY")
    # If no API key, fall back to mock mode so Streamlit app can run without a paid key
    if not api_key:
        st.info("未偵測到 OPENAI_API_KEY，啟用 Mock 模式（模擬回應）。若要使用真實模型請在 Streamlit Secrets 中設定 OPENAI_API_KEY。")
        out = mock_response(user_input, style, tone)
    else:
        with st.spinner("正在聯絡模型..."):
            try:
                out = call_model(combined, user_input, model=model, temperature=temperature, api_key=api_key)
            except OpenAIError as e:
                # Provide actionable troubleshooting steps without leaking internal details
                st.error("OpenAI API 發生錯誤：無法完成請求。")
                st.markdown("**常見解決方法**：\n- 檢查 `OPENAI_API_KEY` 是否設定正確且未過期\n- 確認帳戶有足夠的餘額或配額\n- 檢查 `model` 參數是否為有效的模型名稱（或嘗試使用 `gpt-4o`/`gpt-4o-mini`）\n- 網路或防火牆可能阻擋至 api.openai.com 的存取")
                st.write(f"錯誤摘要：{str(e)}")
                st.stop()
            except Exception as e:
                st.error("發生非預期的錯誤：\n" + str(e))
                st.stop()

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
