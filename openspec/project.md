# Project Context

## Purpose

建立一個簡單、可重現的機器學習解決方案，用於分類訊息/郵件是否為垃圾郵件（AIoT-DA2025 HW3）。提供基礎流程、清晰的評估指標，以及便於本地訓練和推論的工作流程。

## Tech Stack

### 主要開發環境
- **Python**: 3.9–3.12 (Windows)
- **開發工具**: Jupyter Notebooks（存放於 `sources/`）
- **PowerShell**: Windows 友好的腳本命令

### 核心函式庫
- **機器學習**: scikit-learn
- **資料處理**: pandas, numpy, scipy
- **模型儲存**: joblib
- **文本處理**: nltk（可選，用於 notebooks 中的分詞）

### 專案管理
- **Node.js**: LTS 版本
- **OpenSpec**: CLI 工具，用於規格導向開發
## Project Conventions

### Code Style
- 遵循 **PEP8** 規範
- 優先使用 **type hints**
- 保持函式精簡
- 避免隱藏的全域變數

### CLI Scripts
- 使用 `argparse` 並提供 `--help` 說明
- 明確的 `--input/--output` 路徑參數
- 使用 `--seed` 確保結果可重現

### Project Layout
```
project_root/
├── datasets/    # 原始 CSV 檔案
│   └── sms_spam_no_header.csv
├── scripts/     # 訓練和推論的 CLI 腳本
├── models/      # 模型檔案 (.joblib)
├── sources/     # Jupyter notebooks
└── openspec/    # 規格和變更提案
```

### 可重現性
- 固定隨機種子（例如：42）
- 在 `requirements.txt` 中記錄套件版本
## Architecture Patterns

### 開發哲學
- 輕量化：優先使用單檔案腳本，避免複雜框架
- 模組化：訓練和預測腳本分離
- 模型保存：使用 joblib 儲存模型產物

### 技術選擇
- 特徵工程：TF-IDF vectorizer
- 基礎模型：線性分類器（Logistic Regression）

## Testing Strategy

### 評估方法
- 資料分割：80/20 訓練/測試集
- 固定隨機種子確保可重現性
- 報告指標：
  - Accuracy（準確率）
  - Precision（精確度）
  - Recall（召回率）
  - F1 Score

### 測試層級
- **快速測試**：對已知的垃圾/正常郵件樣本進行預測
- **單元測試**：針對文本清理/分詞等功能（可選）

## Git Workflow

### 分支策略
- `main`: 穩定版本分支
- `feature/*`: 功能開發分支（可選）

### 變更管理
- 使用 OpenSpec 提出變更（`openspec/changes/<change-id>/`）

### Commit 規範
- 簡短、指令式的訊息
- 需要時引用變更 ID
- 範例：`feat(spam-classifier): add training script [add-spam-email-classifier]`

## Domain Context

### 專案目標
- 二元分類問題：spam vs ham

### 常見挑戰
- 類別不平衡
- 小數據集過擬合風險
- 雜訊處理（URLs、數字等）

### 資料集
- 主要：`datasets/sms_spam_no_header.csv`
- 其他：`datasets/` 目錄下的補充資料

## Important Constraints

### 運算需求
- 僅使用 CPU 運算
- 訓練時間控制在 2 分鐘內
- 低記憶體佔用

### 執行環境
- 支援離線運行
- 不需要外部 API 呼叫
- 可重現的結果（使用固定種子）
- 適合評分和驗證

## External Dependencies

### 開發工具
- Node.js + OpenSpec CLI（全域安裝）
  - 用於管理規格和變更

### Python 套件
- **scikit-learn**: 機器學習核心
- **pandas**: 資料處理
- **numpy**: 數值計算
- **scipy**: 科學計算
- **joblib**: 模型序列化
