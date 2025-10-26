# 垃圾郵件分類器基礎功能提案

## Why
需要一個簡單且可靠的垃圾郵件分類系統，作為 AIoT-DA2025 HW3 的基礎實作。此系統應該能夠有效區分垃圾郵件（spam）和正常郵件（ham），並提供清晰的評估指標。

## What Changes
- 實作基礎的文本預處理流程
- 建立 TF-IDF 特徵提取流程
- 實作 Logistic Regression 分類器
- 提供模型評估與效能報告
- 建立簡單的命令列介面

## Impact
### 影響的規格
- 新增 `specs/spam-detection/spec.md`：定義垃圾郵件分類的核心功能

### 影響的程式碼
- `scripts/preprocess.py`：文本清理和預處理
- `scripts/train_classifier.py`：模型訓練
- `scripts/predict.py`：預測腳本
- `requirements.txt`：專案依賴