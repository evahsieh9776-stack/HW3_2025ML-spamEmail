# Spam Email Detection Project

## Demo

您可以在以下連結體驗即時的垃圾郵件分類系統：

[https://hw3spamemail2025.streamlit.app/]

### 演示功能
- 資料集分析與視覺化
- 模型性能評估
- 即時垃圾郵件預測
- 互動式閾值調整
- 文本特徵分析

## 來源參考 / Source Reference

本專案基於 Packt 倉庫第 3 章的垃圾郵件問題相關模式和數據集。我們擴充了預處理步驟並添加了更豐富的視覺化工作。

- 原始倉庫：[Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git)

This professional research report integrates the 2025ML‑spamEmail project with examples from Packt's Hands‑On Artificial Intelligence for Cybersecurity, focusing on Phase 4 — visualizations for a Spam/Ham classifier. It includes dataset description, preprocessing, feature engineering, model comparison, a catalogue of visualizations (how to produce them), Streamlit app structure and deployment instructions (target UI like https://2025spamemail.streamlit.app/), and extension ideas for cybersecurity use cases.

## 目錄 / Table of Contents

- Dataset & Sources / 資料集與來源

- Summary of Earlier Phases / 先前階段摘要

- Visualizations Plan (Phase 4) / 視覺化計畫（階段4）

- Implementation Details / 實作細節（程式碼片段、函式說明）

- Streamlit App — UI & Deployment / Streamlit 應用：UI 與部署

- Evaluation & Threshold Analysis / 評估與閾值分析

- Cybersecurity Integration / 資安整合應用

- Future Work & Recommendations / 未來工作建議

- Appendix: Commands & File Layout / 附錄：指令與檔案結構
##
## 1. Dataset & Sources / 資料集與來源

原始專案：2025ML‑spamEmail（sms_spam_no_header.csv）— 主要為 SMS / 簡訊型垃圾訊息資料，label + text 格式。

補充資源：Packt 的 Hands‑On Artificial Intelligence for Cybersecurity 範例程式碼與資料，包含釣魚郵件、惡意 URL、網路流量樣本，使得系統能擴充到真實 Email/釣魚攻擊檢測。

備註 / Note: 若要處理 Email 而非簡訊，需額外匯入 email header、subject、attachments metadata、embedded URLs 等欄位。

## 2. Summary of Earlier Phases / 先前階段摘要

（簡短列出 Phase1–3，讓視覺化階段有完整上下文）

Phase 1 — Data Ingestion & Cleaning / 資料讀入與清洗

讀取 CSV，處理無標頭情況、UTF‑8 編碼、去除空欄。

文本清理：lowercase、移除非 ASCII（選擇性）、URL/Email/數字/標點處理。

Phase 2 — Feature Engineering / 特徵工程

TF / TF‑IDF 向量化（ngram=(1,2)、min_df=2、sublinear_tf）

額外特徵：字數、詞數、URL 數量、has_html、special_chars_ratio、domain_blacklist_flag

Phase 3 — Model Training & Selection / 模型訓練與選擇

傳統模型：Logistic Regression, SVM (LinearSVC), Random Forest

進階模型選項：DistilBERT/BERT 微調、輕量 Transformer for text classification

評估方式：train/validation/test split 或 stratified k‑fold；使用 class_weight 或重抽樣處理類別不平衡

## 3. Visualizations Plan (Phase 4) / 視覺化計畫（階段4）

以下為 Phase 4 須產出的視覺化清單（每項皆包含目的、簡短技術實作提示、與範例程式碼片段位置）

- 3.1 Class Distribution / 類別分佈

目的：確認資料不平衡程度（Spam vs Ham）。

實作：簡單 bar chart（counts），或百分比圓餅圖。

輸出：plots/class_distribution.png

- 3.2 Top Tokens per Class / 各類別關鍵字頻率（Top N tokens）

目的：了解哪些詞在 spam 與 ham 中高頻出現，提供特徵解讀。可用於黑名單或規則式檢測。

實作：對清洗後文本分別計算 token frequency（或 use CountVectorizer），繪製水平條狀圖 top20。

輸出：plots/top_tokens_spam.png, plots/top_tokens_ham.png

- 3.3 Word Clouds / 詞雲

目的：視覺化高頻詞組直觀印象。

實作：wordcloud lib，分 class 顯示。

- 3.4 Feature Importance / 特徵重要性（模型差異）

目的：解釋模型決策（哪些詞/特徵最具區隔能力）。

實作：

對於線性模型（LogReg/SVM）可取 coef 與 top coefficients；

對於樹模型（RandomForest）使用 feature_importances_；

使用 SHAP 或 LIME 做局部解釋（更進階）。

輸出：plots/feature_importance.png, plots/shap_summary.png

- 3.5 Confusion Matrix / 混淆矩陣

目的：查看 False Positive（誤判為 Spam 的 Ham）與 False Negative（漏判 Spam）的數量。

實作：scikit‑learn confusion_matrix，sns.heatmap 或 matplotlib.imshow。

- 3.6 ROC & PR Curves / ROC 與 Precision‑Recall 曲線

目的：全面評估模型在不同閾值下的分類表現（尤其 class imbalance 下 PR 曲線更重要）。

實作：scikit‑learn roc_curve, precision_recall_curve。

輸出：plots/roc_curve.png, plots/pr_curve.png

- 3.7 Threshold Sweep / 閾值掃描分析

目的：展示不同閾值（probability threshold）下 precision/recall/f1 的 trade‑offs，並在 Streamlit UI 上提供滑桿控制。

實作：迴圈掃描 thresholds=np.linspace(0.0,1.0,101)，計算 metric，繪製三線圖。

輸出：plots/threshold_sweep.png

- 3.8 Time Series / Drift Monitoring（若有時間序列資料）

目的：觀察 spam 比例隨時間變化，偵測模型漂移。

實作：以日期桶（by week/月）繪圖，顯示 spam rate。

- 3.9 Example Predictions with Explanations / 具解釋的範例預測

目的：在 UI 上顯示單筆測試文字、預測機率、模型解釋（關鍵詞或 SHAP force plot）。

實作：選擇若干典型樣本（TP, FP, FN, TN），展示原始文字 + top contributing tokens。

## 4. Implementation Details / 實作細節（含程式碼片段）

以下僅列核心片段與重點指令；完整程式碼建議置於 scripts/visualize_spam.py 中。

4.1 讀取資料與向量器
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('datasets/processed/sms_spam_clean.csv')
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
X = vectorizer.fit_transform(df['text_clean'])
4.2 Top tokens per class
from collections import Counter


def top_tokens_for_class(df, label='spam', topn=20):
    texts = df[df['label']==label]['text_clean']
    tokens = Counter()
    for t in texts:
        tokens.update(t.split())
    return tokens.most_common(topn)
4.3 Feature importance for linear models
import numpy as np


coef = model.coef_.ravel()
feature_names = vectorizer.get_feature_names_out()
top_pos_idx = np.argsort(coef)[-20:][::-1]
top_neg_idx = np.argsort(coef)[:20]
4.4 Confusion matrix & curves
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)


fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
4.5 SHAP explanation snippet
import shap
explainer = shap.Explainer(model.predict, vectorizer.transform)
shap_values = explainer(X_sample)
shap.summary_plot(shap_values, feature_names=feature_names)
## 5. Streamlit App — UI & Deployment / Streamlit 應用：UI 與部署

目標：建立一個類似 https://2025spamemail.streamlit.app/ 的互動頁面，包含單筆輸入、批次上傳、可視化面板、閾值滑桿、以及模型解釋區。

5.1 建議頁面佈局（Sidebar + Main）

Sidebar：模型選擇下拉、閾值滑桿（0–1）、上傳 CSV 按鈕、下載 predictions

Main：

標題與說明

單筆輸入文字框 + 預測按鈕

機率條（progress bar）與分類結果（Spam / Ham）

可視化標籤卡（tabs）：Class Distribution、Top Tokens、ROC/PR、Threshold Sweep

範例預測（TP/FP/FN/TN）並顯示解釋（top contributing tokens / SHAP）

5.2 主要程式骨幹（app.py 範例）
import streamlit as st
import joblib


model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')


st.title('Spam/Ham Classifier — Demo')


threshold = st.sidebar.slider('Decision Threshold', 0.0, 1.0, 0.5)


text = st.text_area('Enter message text')
if st.button('Predict'):
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0,1]
    label = 'Spam' if prob>=threshold else 'Ham'
    st.metric('Prediction', label, delta=f"P(spam)={prob:.3f}")
    # show top tokens and explanation
5.3 部署步驟（Streamlit Cloud）

在 repo 根目錄新增 requirements.txt（列出 streamlit、scikit‑learn、joblib、shap、matplotlib、pandas、wordcloud 等）。

將 app.py push 到 GitHub；進入 Streamlit Cloud（或其他 host），連結該 repo。

設定啟動命令 streamlit run app.py。

若需私有模型或大檔案，建議使用 Git LFS 或將模型放在 object storage 並在啟動時下載。

## 6. Evaluation & Threshold Analysis / 評估與閾值分析

關鍵指標：Precision / Recall / F1 / AUC‑ROC / PR‑AUC

重點：在 spam 偵測任務中常以 Recall（偵測 spam 的能力）與 Precision（降低誤判為 spam 的成本）做 trade‑off。實務上可指定 target precision 或 recall，然後選閾值。

示例：

若業務目標是「盡量不要漏掉 spam（高 recall）」，可設閾值較低（0.3~0.5）；

若業務目標是「盡量減少誤判（高 precision）」，閾值設高（0.6~0.9）。

- 在 UI 上建議加入「建議閾值（based on target metric）」的按鈕，幫使用者自動選閾值。

## 7. Cybersecurity Integration / 資安整合應用

釣魚郵件偵測（Phishing Detection）：結合 URL 分析（特徵：punycode, domain age, IP in URL, blacklisted domains）與文本特徵。

附件風險評分：若能攔截附件，使用靜態分析（hash match）與 sandbox 行為分數作為附加特徵。

網路層警示整合：將分類器結合 SIEM（Splunk/ELK）或郵件閘道，若預測為 spam 或 high risk, 自動 append header 或 quarantine。

對抗樣本防護：研究 adversarial training、input sanitization 以因應詭計性文本改寫（obfuscation）攻擊。

## 8. Future Work & Recommendations / 未來工作與建議

引入 Transformer 微調（BERT/DistilBERT）：比較在短文本（SMS）與 Email（長文本）上的效益。

多模態特徵：結合 header、URL、附件、連結行為與用戶互動特徵（點擊率、回覆率）。

模型監控：部署後建立 drift detection 與自動 re‑training pipeline（如每月/事件驅動）。

可解釋性：在 UI 中嵌入 SHAP force plots，提供給資安分析師更好判斷。

部署硬化：API 安全、模型權限管理、輸入速率限制（rate limiting）。

## 9. Appendix: Commands & File Layout / 附錄：指令與檔案結構

推薦檔案結構

repo-root/
├─ datasets/
│  ├─ sms_spam_no_header.csv
│  └─ processed/
├─ models/
│  ├─ spam_classifier.pkl
│  └─ vectorizer.pkl
├─ scripts/
│  ├─ preprocess_emails.py
│  ├─ train_spam_classifier.py
│  └─ visualize_spam.py
├─ app.py
├─ requirements.txt
└─ README.md

常用指令

python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1
python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col label --text-col text_clean
python scripts/visualize_spam.py --input datasets/processed/sms_spam_clean.csv --outdir plots/
streamlit run app.py