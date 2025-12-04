# 🚀 快速入門指南

## 快速開始 (5分鐘)

### 1️⃣ 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 或使用conda
conda env create -f environment.yml
```

### 2️⃣ 準備數據

將八哥鳥類圖像放在 `mynah_data` 目錄下：

```
mynah_data/
├── mynah/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── other_class/
    ├── img_101.jpg
    └── ...
```

### 3️⃣ 運行實驗

#### 使用 Jupyter Notebook (推薦)
```bash
jupyter notebook hw4.ipynb
```

#### 使用 Python 腳本
```bash
python hw4.py
```

### 4️⃣ 查看結果

所有結果保存在 `results/` 目錄：
- 📊 可視化圖表
- 📈 性能指標
- 📝 研究報告

---

## 📊 項目結構詳解

### Notebook 單元說明

| 單元 | 功能 |
|------|------|
| 1 | 導入庫和環境設置 |
| 2 | 數據加載與探索 |
| 3 | 定義5種增強策略 |
| 4 | 創建數據加載器 |
| 5 | 構建ResNet18模型 |
| 6 | 訓練所有模型 |
| 7 | 測試集評估 |
| 8 | 訓練曲線可視化 |
| 9 | 混淆矩陣分析 |
| 10 | 增強效果展示 |
| 11 | 統計分析與結論 |
| 12 | 成果總結 |

---

## 🔧 常見問題

### Q1: 如何修改訓練參數？

編輯 Notebook 中的以下單元格或修改 `config.json`:

```python
EPOCHS = 50          # 訓練次數
LEARNING_RATE = 0.001  # 學習率
BATCH_SIZE = 32      # 批大小
```

### Q2: 如何使用自己的數據集？

1. 將數據放在 `mynah_data/` 目錄
2. 修改 `load_dataset()` 函數中的 `data_dir` 參數
3. 確保目錄結構為: `data_dir/類別名/圖像.jpg`

### Q3: 如何加載已訓練的模型？

```python
model = build_model(num_classes=2)
model.load_state_dict(torch.load('models/Combined_best.pth'))
model.eval()

# 進行預測
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.softmax(output, dim=1)
```

### Q4: 如何調整增強參數？

編輯 `create_augmentation_strategies()` 函數：

```python
'Geometric': transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.7),  # 修改概率
    transforms.RandomRotation(degrees=45),  # 修改旋轉角度
    ...
])
```

### Q5: 如何在GPU上訓練？

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")
```

---

## 📈 性能優化建議

### 1. 加速訓練
```python
# 增加 num_workers
DataLoader(..., num_workers=4, pin_memory=True)

# 使用混合精度訓練
from torch.cuda.amp import autocast
with autocast():
    outputs = model(images)
```

### 2. 改善準確率
```python
# 調整超參數
LEARNING_RATE = 0.0005  # 降低學習率
WEIGHT_DECAY = 5e-4     # 增加正則化

# 使用更多增強
# 或調整增強強度
```

### 3. 減少過擬合
```python
# 增加 Dropout
nn.Dropout(0.7)  # 提高 dropout 比率

# 使用早停
PATIENCE = 5  # 較早停止

# 增加數據增強強度
```

---

## 🎯 實驗對比

### 如何對比不同設置？

修改並運行多次，比較結果：

```python
# 方案A: 使用 Combined 增強
strategy = 'Combined'
model_a = train_model(...)

# 方案B: 使用 Occlusion 增強
strategy = 'Occlusion'
model_b = train_model(...)

# 對比性能
print("方案A準確率:", metrics_a['accuracy'])
print("方案B準確率:", metrics_b['accuracy'])
```

---

## 📚 進階用法

### 自定義增強策略

```python
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    normalize
])
```

### 自定義模型架構

```python
class CustomMynahClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
```

### 導出模型到 ONNX

```python
import torch.onnx
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## 💾 保存和加載檢查點

```python
# 保存檢查點
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加載檢查點
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 📞 支持與反饋

遇到問題？
1. 查看 Notebook 中的錯誤信息
2. 檢查數據格式是否正確
3. 確認所有依賴已正確安裝

---

**Happy Training! 🎉**
