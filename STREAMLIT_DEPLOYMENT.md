# 🚀 Streamlit 應用部署指南

## 📋 目錄
1. [本地運行](#本地運行)
2. [Streamlit Cloud 部署](#streamlit-cloud-部署)
3. [Heroku 部署](#heroku-部署)
4. [Docker 部署](#docker-部署)
5. [常見問題](#常見問題)

---

## 🏠 本地運行

### 前置要求
- Python 3.8+
- pip 或 conda

### 安裝步驟

#### 1. 安裝 Streamlit
```bash
pip install streamlit streamlit-option-menu
```

#### 2. 安裝依賴包
```bash
pip install pandas numpy matplotlib seaborn plotly pillow scikit-learn
```

或一次性安裝：
```bash
pip install -r requirements.txt
```

#### 3. 運行應用
```bash
streamlit run streamlit_app.py
```

#### 4. 訪問應用
應用將在 `http://localhost:8501` 上運行

### 本地運行的優勢
- 完全離線運行
- 支持快速開發和調試
- 無需依賴遠程服務
- 適合本地測試

---

## ☁️ Streamlit Cloud 部署 (推薦)

Streamlit Cloud 是官方的免費部署平台。

### 部署步驟

#### 1. 準備 GitHub 倉庫
```bash
# 初始化 Git 倉庫
git init

# 添加文件
git add .

# 提交
git commit -m "Initial commit: Streamlit app"

# 推送到 GitHub
git remote add origin https://github.com/你的用戶名/你的倉庫.git
git push -u origin main
```

#### 2. 創建 `requirements.txt`
```
streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
scikit-learn==1.2.0
Pillow==9.5.0
```

#### 3. 在 Streamlit Cloud 上部署

1. 訪問 [Streamlit Cloud](https://share.streamlit.io)
2. 使用 GitHub 帳號登錄
3. 點擊 "New app"
4. 選擇倉庫、分支和文件
5. 點擊 "Deploy"

#### 4. 監控應用
- 查看日誌
- 管理密鑰和secrets
- 設置環境變量

### Streamlit Cloud 優勢
✅ 完全免費
✅ 一鍵部署
✅ 自動SSL證書
✅ 自動扩展
✅ GitHub 集成
✅ 無需管理服務器

---

## 🚀 Heroku 部署

### 前置要求
- Heroku 帳號
- Heroku CLI

### 部署步驟

#### 1. 安裝 Heroku CLI
```bash
# Windows
choco install heroku-cli

# Mac
brew tap heroku/brew && brew install heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

#### 2. 登錄 Heroku
```bash
heroku login
```

#### 3. 創建 `Procfile`
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### 4. 創建 `setup.sh`
```bash
mkdir -p ~/.streamlit/
echo "[theme]
primaryColor = '#667eea'
backgroundColor = '#f8f9fa'
secondaryBackgroundColor = '#ffffff'
textColor = '#2c3e50'
font = 'sans serif'

[client]
showErrorDetails = true

[server]
port = \$PORT
enableCORS = false
headless = true" > ~/.streamlit/config.toml
```

#### 5. 部署
```bash
# 創建 Heroku 應用
heroku create 你的應用名稱

# 推送代碼
git push heroku main

# 查看日誌
heroku logs --tail
```

#### 6. 訪問應用
```
https://你的應用名稱.herokuapp.com
```

---

## 🐳 Docker 部署

### 創建 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用文件
COPY . .

# 暴露端口
EXPOSE 8501

# 運行應用
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 構建和運行

#### 1. 構建 Docker 鏡像
```bash
docker build -t mynah-app .
```

#### 2. 運行容器
```bash
docker run -p 8501:8501 mynah-app
```

#### 3. 訪問應用
```
http://localhost:8501
```

### Docker Compose (可選)

創建 `docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - .:/app
```

運行：
```bash
docker-compose up
```

---

## 🌐 在線部署平台比較

| 平台 | 成本 | 難度 | 特點 | 推薦 |
|------|------|------|------|------|
| **Streamlit Cloud** | 免費 | ⭐ | 一鍵部署，GitHub集成 | ✅ 首選 |
| **Heroku** | 免費→付費 | ⭐⭐ | 靈活配置，支持多種方式 | ✅ 中等 |
| **PythonAnywhere** | 免費→付費 | ⭐⭐ | Python友好，易於部署 | ✅ 備選 |
| **AWS** | 付費 | ⭐⭐⭐ | 高性能，完全控制 | 大型應用 |
| **Google Cloud** | 付費 | ⭐⭐⭐ | 企業級，多種服務 | 大型應用 |
| **Azure** | 付費 | ⭐⭐⭐ | 企業級，集成度高 | 企業用戶 |

---

## ⚙️ 性能優化

### 1. 緩存數據
```python
@st.cache_data
def load_data():
    # 加載數據
    return data
```

### 2. 會話管理
```python
import streamlit as st

if 'counter' not in st.session_state:
    st.session_state.counter = 0
```

### 3. 圖表優化
```python
# 使用 Plotly 而不是 Matplotlib
# Plotly 更輕量級，加載更快
```

### 4. 資源優化
```python
# 限制圖像大小
# 使用增量加載
# 優化查詢
```

---

## 🔒 安全性

### 1. secrets.toml
創建 `.streamlit/secrets.toml`:
```toml
[database]
host = "xxx"
user = "xxx"
password = "xxx"

[api]
key = "xxx"
token = "xxx"
```

訪問：
```python
db_password = st.secrets["database"]["password"]
api_key = st.secrets["api"]["key"]
```

### 2. 環境變量
```bash
# 設置環境變量
export STREAMLIT_SERVER_PORT=8501

# 在應用中使用
import os
port = os.getenv('STREAMLIT_SERVER_PORT', 8501)
```

### 3. 身份驗證
```python
import streamlit as st

def check_password():
    if st.secrets.get("password") is None:
        st.error("Password not found in secrets")
        return False
    
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True
    
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    return False

if not check_password():
    st.stop()
```

---

## 📊 監控和日誌

### 1. Streamlit 日誌
```bash
# 查看本地日誌
streamlit logs

# 設置日誌級別
streamlit run app.py --logger.level=debug
```

### 2. 自定義日誌
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("應用已啟動")
```

### 3. 性能監控
```python
import time

start = time.time()
# 執行操作
end = time.time()
st.write(f"耗時: {end - start:.2f} 秒")
```

---

## 🐛 常見問題

### Q1: 應用加載很慢？
**A:** 
- 使用 `@st.cache_data` 緩存數據
- 優化圖表渲染
- 減少頁面元素
- 使用 CDN 加載資源

### Q2: 如何更新已部署的應用？
**A:**
- Streamlit Cloud: 自動更新 (GitHub push)
- Heroku: `git push heroku main`
- Docker: 重新構建和推送鏡像

### Q3: 環境變量如何設置？
**A:**
- Streamlit Cloud: 在設置中添加 secrets
- Heroku: `heroku config:set KEY=VALUE`
- Docker: 環境變量或 .env 文件

### Q4: 如何處理大文件上傳？
**A:**
```python
uploaded_file = st.file_uploader("上傳文件", type=['csv', 'xlsx'])
if uploaded_file is not None:
    if uploaded_file.size > 100 * 1024 * 1024:  # 100MB
        st.error("文件過大")
    else:
        # 處理文件
```

### Q5: 如何自定義域名？
**A:**
- Streamlit Cloud: 購買域名，配置 CNAME
- Heroku: 購買域名，添加到應用
- 自托管: 使用 Nginx 反向代理

---

## 📚 有用資源

- [Streamlit 官方文檔](https://docs.streamlit.io)
- [Streamlit Cloud 文檔](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit 部署指南](https://docs.streamlit.io/library/get-started/installation)
- [Streamlit 社區](https://discuss.streamlit.io)
- [Streamlit 應用庫](https://streamlit.io/gallery)

---

## 🎯 推薦部署方案

### 對於初學者
✅ **Streamlit Cloud** - 最簡單，推薦首選

### 對於個人項目
✅ **Streamlit Cloud** 或 **Heroku** - 免費且穩定

### 對於小團隊
✅ **Heroku** 或 **PythonAnywhere** - 付費選項合理

### 對於企業應用
✅ **AWS** 或 **Google Cloud** - 企業級功能

---

## 🚀 快速開始

### 最簡單的方式 (Streamlit Cloud)

```bash
# 1. 準備代碼
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用戶名/倉庫.git
git push -u origin main

# 2. 訪問 https://share.streamlit.io
# 3. 連接 GitHub 並部署
# 4. 完成！應用已上線
```

---

**🎉 祝您部署成功！**

如有任何問題，請參考 [官方文檔](https://docs.streamlit.io) 或 [社區論壇](https://discuss.streamlit.io)。
