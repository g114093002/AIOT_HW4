"""Streamlit app for 員瑛式思考生成器 — deployable with `streamlit run app_streamlit.py`"""
import os
import json

import streamlit as st
from openai import OpenAI

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


def call_model(system_prompt: str, user_input: str, model: str, temperature: float):
    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


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
    with st.spinner("正在聯絡模型..."):
        out = call_model(combined, user_input, model=model, temperature=temperature)
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
