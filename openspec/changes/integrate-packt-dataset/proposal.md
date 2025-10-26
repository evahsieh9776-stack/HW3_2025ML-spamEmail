# 整合 Packt 資料集與範例

## Why
為了提供更豐富的訓練資料和參考實作，我們需要整合 Packt 的 "Hands-On-Artificial-Intelligence-for-Cybersecurity" 專案中的垃圾郵件分類範例。這將幫助我們：
1. 獲得額外的訓練資料
2. 參考專業的實作方法
3. 改進現有的分類器設計

## What Changes
- 新增資料集下載和處理腳本
- 整合 Packt 範例的特徵工程方法
- 採用其預處理步驟
- 擴展模型評估方法
- 改進視覺化功能

## Impact

### 影響的規格
- 修改 `specs/spam-detection/spec.md`：
  - 新增資料集整合需求
  - 擴展特徵工程部分
  - 增加視覺化需求

### 影響的程式碼
- 新增 `scripts/download_dataset.py`
- 新增 `scripts/data_integration.py`
- 修改 `scripts/preprocess.py`
- 修改 `requirements.txt`