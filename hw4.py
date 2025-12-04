"""
資料增強對八哥辨識模型的影響分析
Data Augmentation Impact on Mynah Bird Classification using Transfer Learning

本模塊提供了完整的數據增強和遷移學習實現框架。
可以獨立運行此文件或導入到 Jupyter Notebook 中使用。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import RandomErasing

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

# ============================================================================
# 配置常數
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10

# 隨機種子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ============================================================================
# 自定義數據集類
# ============================================================================

class MynahDataset(Dataset):
    """八哥鳥圖像數據集類"""
    
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, label


# ============================================================================
# 數據加載和增強定義
# ============================================================================

def create_augmentation_strategies():
    """創建5種不同的數據增強策略"""
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    strategies = {
        'Baseline': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ]),
        
        'Geometric': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            normalize
        ]),
        
        'Color': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ]),
        
        'Combined': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ]),
        
        'Occlusion': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ]),
    }
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    return strategies, test_transform


def load_dataset(data_dir, test_size=0.2, val_size=0.1):
    """加載數據集"""
    
    image_paths = []
    labels = []
    class_names = []
    
    data_dir = Path(data_dir)
    
    if data_dir.exists():
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                class_idx = len(class_names) - 1
                
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_file in class_dir.glob(ext):
                        image_paths.append(str(img_file))
                        labels.append(class_idx)
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    n_samples = len(image_paths)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_count = int(n_samples * test_size)
    val_count = int(n_samples * val_size)
    train_count = n_samples - test_count - val_count
    
    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]
    
    return image_paths, labels, class_names, train_idx, val_idx, test_idx


def create_dataloaders(image_paths, labels, train_idx, val_idx, test_idx,
                       augmentation_strategy, batch_size=BATCH_SIZE):
    """創建數據加載器"""
    
    _, test_transform = create_augmentation_strategies()
    
    train_dataset = MynahDataset(
        image_paths[train_idx],
        labels[train_idx],
        transforms=augmentation_strategy
    )
    
    val_dataset = MynahDataset(
        image_paths[val_idx],
        labels[val_idx],
        transforms=test_transform
    )
    
    test_dataset = MynahDataset(
        image_paths[test_idx],
        labels[test_idx],
        transforms=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 模型相關函數
# ============================================================================

def build_model(num_classes):
    """構建ResNet18遷移學習模型"""
    
    model = models.resnet18(pretrained=True)
    
    # 凍結早期層
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # 修改最後的全連接層
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


def count_parameters(model):
    """計算模型參數"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# 訓練相關函數
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """驗證模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, test_loader, device):
    """評估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# 主程序
# ============================================================================

def main(data_dir='mynah_data', results_dir='results', models_dir='models'):
    """主程序"""
    
    # 創建目錄
    Path(results_dir).mkdir(exist_ok=True)
    Path(models_dir).mkdir(exist_ok=True)
    
    print("="*80)
    print("資料增強對八哥辨識模型的影響分析")
    print("="*80)
    
    # 加載數據
    print("\n📂 正在加載數據...")
    try:
        image_paths, labels, class_names, train_idx, val_idx, test_idx = \
            load_dataset(data_dir)
    except Exception as e:
        print(f"❌ 加載數據失敗: {e}")
        return
    
    num_classes = len(class_names)
    print(f"✓ 已加載 {len(image_paths)} 個圖像，{num_classes} 個類別")
    
    # 創建增強策略
    strategies, test_transform = create_augmentation_strategies()
    print(f"✓ 已定義 {len(strategies)} 種增強策略")
    
    # 訓練所有模型
    all_histories = {}
    trained_models = {}
    
    for strategy_name in strategies.keys():
        print(f"\n{'='*60}")
        print(f"訓練 {strategy_name} 模型")
        print(f"{'='*60}")
        
        # 創建數據加載器
        train_loader, val_loader, test_loader = create_dataloaders(
            image_paths, labels, train_idx, val_idx, test_idx,
            strategies[strategy_name]
        )
        
        # 構建模型
        model = build_model(num_classes).to(DEVICE)
        
        # 定義優化器和損失函數
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 訓練
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader,
                                               criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                      f"Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(),
                          Path(models_dir) / f'{strategy_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"早停: 驗證準確率 {PATIENCE} 個 epoch 未改進")
                    break
        
        # 加載最佳模型
        model.load_state_dict(torch.load(Path(models_dir) / f'{strategy_name}_best.pth'))
        all_histories[strategy_name] = history
        trained_models[strategy_name] = model
        
        # 評估
        y_pred, y_true = evaluate_model(model, test_loader, DEVICE)
        test_acc = accuracy_score(y_true, y_pred)
        print(f"✓ 測試準確率: {test_acc:.4f}")
    
    print(f"\n✅ 所有 {len(strategies)} 個模型訓練完成")
    print(f"💾 模型已保存至: {models_dir}")
    print(f"📊 結果已保存至: {results_dir}")


if __name__ == '__main__':
    # 檢查數據目錄
    data_dir = 'mynah_data'
    if not Path(data_dir).exists():
        print(f"❌ 錯誤: 數據目錄 '{data_dir}' 不存在")
        print(f"請將數據放在 {Path.cwd() / data_dir} 目錄下")
        print("數據格式: mynah_data/類別名1/image1.jpg, mynah_data/類別名2/image2.jpg")
        sys.exit(1)
    
    # 運行主程序
    main(data_dir='mynah_data', results_dir='results', models_dir='models')
