"""
Spam Email Classification Visualization App
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import re

# 設置頁面配置
st.set_page_config(page_title="Spam Email Classification", layout="wide")

# 添加主標題
st.title("Spam/Ham Classifier — Phase 4 Visualizations")

# 側邊欄配置
st.sidebar.title("模型配置")

# 數據集選擇
dataset_path = st.sidebar.selectbox(
    "Dataset CSV",
    ["datasets/raw/sms_spam_no_header.csv"],
    index=0
)

# 列選擇
label_col = st.sidebar.selectbox(
    "Label column",
    ["ham"],  # 第一列為標籤
    index=0
)

text_col = st.sidebar.selectbox(
    "Text column",
    ["message"],  # 第二列為文本
    index=0
)

# 模型目錄
models_dir = st.sidebar.selectbox(
    "Models dir",
    ["models"],
    index=0
)

# 其他參數
seed = st.sidebar.number_input("Seed", value=42)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

# 載入數據
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['label', 'message']  # 重命名列
    return df

# 數據清理和預處理
@st.cache_data
def preprocess_text(text):
    # 轉換為小寫
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 移除多餘空白
    text = ' '.join(text.split())
    return text

# 主要內容
def main():
    # 載入數據
    df = load_data(dataset_path)
    
    # 數據預處理
    df['cleaned_text'] = df['message'].apply(preprocess_text)
    
    # 分割訓練和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        (df['label'] == 'spam').astype(int),
        test_size=test_size,
        random_state=seed
    )
    
    # 特徵提取
    vectorizer = CountVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 載入模型
    model = joblib.load(Path(models_dir) / 'spam_classifier.joblib')
    
    # 預測概率
    y_pred_proba = model.decision_function(X_test_vec)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # 創建兩列佈局
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Data Overview")
        
        # 類別分佈
        fig_dist = px.pie(
            names=['Ham', 'Spam'],
            values=df['label'].value_counts(),
            title="Class Distribution"
        )
        st.plotly_chart(fig_dist)
        
        # Token replacements
        st.subheader("Token replacements in cleaned text (approximate)")
        total_tokens = sum(len(text.split()) for text in df['cleaned_text'])
        st.write(f"Total tokens: {total_tokens}")
        
    with col2:
        st.header("Top Tokens by Class")
        
        # 添加Top-N滑桿
        top_n = st.slider("Top-N tokens", min_value=5, max_value=50, value=10, step=5)
        
        # 計算每個類別的高頻詞
        spam_texts = ' '.join(df[df['label'] == 'spam']['cleaned_text'])
        ham_texts = ' '.join(df[df['label'] == 'ham']['cleaned_text'])
        
        spam_words = Counter(spam_texts.split()).most_common(top_n)
        ham_words = Counter(ham_texts.split()).most_common(top_n)
        
        # 創建條形圖
        fig_tokens = go.Figure()
        
        # 添加垃圾郵件詞頻
        fig_tokens.add_trace(go.Bar(
            x=[word[0] for word in spam_words],
            y=[word[1] for word in spam_words],
            name='Spam',
            marker_color='red'
        ))
        
        # 添加正常郵件詞頻
        fig_tokens.add_trace(go.Bar(
            x=[word[0] for word in ham_words],
            y=[word[1] for word in ham_words],
            name='Ham',
            marker_color='blue'
        ))
        
        fig_tokens.update_layout(
            title="Top-10 tokens by class",
            barmode='group',
            xaxis_title="Token",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_tokens)
    
    # 模型性能評估
    st.header("Model Performance (Test)")
    
    # 計算各種性能指標
    from sklearn.metrics import (confusion_matrix, classification_report, 
                               roc_curve, auc, precision_recall_curve)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 創建性能指標表格
    metrics_data = {
        "Metric": [
            "Accuracy", "Precision", "Recall", "F1-Score",
            "True Negatives", "False Positives", 
            "False Negatives", "True Positives"
        ],
        "Value": [
            (tp + tn) / (tp + tn + fp + fn),
            tp / (tp + fp) if (tp + fp) > 0 else 0,
            tp / (tp + fn) if (tp + fn) > 0 else 0,
            2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            tn, fp, fn, tp
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # 顯示性能指標表格
    st.subheader("Performance Metrics")
    # 將數值列轉換為 Python 原生類型
    metrics_df['Value'] = metrics_df.apply(
        lambda row: float(row['Value']) if row.name < 4 else int(row['Value']),
        axis=1
    )
    
    # 應用格式化
    formatted_df = metrics_df.copy()
    formatted_df['Value'] = formatted_df.apply(
        lambda row: f"{row['Value']:.4f}" if row.name < 4 else f"{int(row['Value']):d}",
        axis=1
    )
    st.dataframe(formatted_df)
    
    # 創建兩列佈局用於顯示圖表
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # 混淆矩陣
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Ham', 'Spam'],
            y=['Ham', 'Spam'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False,
            colorscale='RdBu',
            showscale=True
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            xaxis=dict(side="bottom"),
            width=400,
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with plot_col2:
        # ROC 曲線
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            mode='lines'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Precision-Recall 曲線
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision,
        name=f'PR curve (AUC = {pr_auc:.3f})',
        mode='lines'
    ))
    fig_pr.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.05])
    )
    st.plotly_chart(fig_pr)
    
    # 閾值掃描
    st.header("Threshold sweep (precision/recall/f1)")
    
    # 讓使用者選擇閾值範圍和點數
    thresh_col1, thresh_col2, thresh_col3 = st.columns(3)
    with thresh_col1:
        thresh_min = st.number_input("最小閾值", value=-5.0, step=0.5)
    with thresh_col2:
        thresh_max = st.number_input("最大閾值", value=5.0, step=0.5)
    with thresh_col3:
        thresh_points = st.number_input("取樣點數", value=21, min_value=5, max_value=50)
    
    # 計算不同閾值下的性能指標
    thresholds_range = np.linspace(thresh_min, thresh_max, thresh_points)
    threshold_metrics = []
    
    for thresh in thresholds_range:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # 計算MCC (Matthews Correlation Coefficient)
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 1
        mcc = mcc_numerator / mcc_denominator
        
        threshold_metrics.append({
            'Threshold': thresh,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Specificity': specificity,
            'NPV': npv,
            'MCC': mcc,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        })
    
    # 創建閾值性能表格
    thresh_df = pd.DataFrame(threshold_metrics)
    
    # 選擇指標的顯示方式
    metric_display = st.multiselect(
        "選擇要顯示的性能指標",
        options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'NPV', 'MCC', 'TP', 'TN', 'FP', 'FN'],
        default=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
    )
    
    # 選擇顏色方案
    color_scheme = st.selectbox(
        "選擇表格配色方案",
        options=['YlOrRd', 'RdYlBu', 'viridis', 'plasma', 'magma'],
        index=0
    )
    
    # 顯示閾值性能表格
    st.subheader("Threshold Performance Metrics")
    
    # 準備要顯示的列
    display_columns = ['Threshold'] + metric_display
    display_df = thresh_df[display_columns].copy()
    
    # 將所有列轉換為 Python 原生類型
    for col in display_df.columns:
        if col in ['TP', 'TN', 'FP', 'FN']:
            display_df[col] = display_df[col].astype(int)
        else:
            display_df[col] = display_df[col].astype(float)
    
    # 創建格式化後的數據框
    formatted_df = display_df.copy()
    for col in formatted_df.columns:
        if col == 'Threshold':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
        elif col in ['TP', 'TN', 'FP', 'FN']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x)}")
        else:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    
    # 找出最佳F1分數的索引
    best_f1_idx = thresh_df['F1-Score'].idxmax()
    
    # 創建樣式函數
    def color_background(s):
        if s.name == best_f1_idx:
            return ['background-color: lightblue'] * len(s)
        return [''] * len(s)
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    # 應用樣式
    styled_df = formatted_df.style\
        .apply(color_background, axis=1)\
        .apply(highlight_max, subset=metric_display)
    
    if color_scheme:
        styled_df = styled_df.background_gradient(subset=metric_display, cmap=color_scheme)
    
    st.dataframe(styled_df)
    
    # 閾值掃描圖
    st.subheader("Threshold Sweep Plot")
    
    # 選擇要在圖表中顯示的指標
    plot_metrics = st.multiselect(
        "選擇要在圖表中顯示的指標",
        options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'NPV', 'MCC'],
        default=['Accuracy', 'Precision', 'Recall', 'F1-Score']
    )
    
    # 定義每個指標的顏色
    metric_colors = {
        'Accuracy': 'purple',
        'Precision': 'blue',
        'Recall': 'red',
        'F1-Score': 'green',
        'Specificity': 'orange',
        'NPV': 'brown',
        'MCC': 'pink'
    }
    
    fig_thresh = go.Figure()
    
    # 添加選擇的指標曲線
    for metric in plot_metrics:
        fig_thresh.add_trace(go.Scatter(
            x=thresholds_range,
            y=thresh_df[metric],
            name=metric,
            line=dict(color=metric_colors[metric])
        ))
    
    # 找出最佳F1分數的閾值
    best_f1_thresh = thresh_df.loc[best_f1_idx, 'Threshold']
    
    fig_thresh.update_layout(
        title="Metrics vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[-0.05, 1.05]),
        xaxis=dict(range=[thresh_min, thresh_max]),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # 添加當前選擇的閾值的垂直線
    fig_thresh.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current threshold: {threshold:.3f}",
        annotation_position="top"
    )
    
    # 添加最佳F1分數的閾值垂直線
    fig_thresh.add_vline(
        x=best_f1_thresh,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Best F1 threshold: {best_f1_thresh:.3f}",
        annotation_position="bottom"
    )
    
    st.plotly_chart(fig_thresh)
    
    # 添加預測部分
    st.header("預測新訊息")
    
    # 文字輸入框
    input_message = st.text_area(
        "輸入要檢測的訊息",
        height=100,
        placeholder="在此輸入要檢測的訊息..."
    )
    
    # 預測按鈕
    if st.button("進行預測"):
        if input_message:
            # 預處理輸入文本
            cleaned_input = preprocess_text(input_message)
            
            # 轉換為特徵向量
            input_vec = vectorizer.transform([cleaned_input])
            
            # 預測
            pred_proba = model.decision_function(input_vec)
            prediction = "垃圾郵件" if pred_proba > threshold else "正常郵件"
            confidence = abs(pred_proba)
            
            # 顯示結果
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.info(f"預測結果：{prediction}")
            
            with result_col2:
                st.info(f"置信度：{confidence:.2f}")
                
            # 顯示文本分析
            st.subheader("文本分析")
            
            # 取得前10個最重要的詞
            feature_names = np.array(vectorizer.get_feature_names_out())
            input_features = input_vec.toarray()[0]
            
            # 找出非零特徵的索引並按重要性排序
            nonzero_idx = input_features.nonzero()[0]
            sorted_idx = nonzero_idx[np.argsort(input_features[nonzero_idx])[::-1]]
            
            # 顯示重要詞彙
            important_words = pd.DataFrame({
                '詞彙': feature_names[sorted_idx[:10]],
                '重要性': input_features[sorted_idx[:10]]
            })
            
            st.write("重要詞彙：")
            st.dataframe(important_words)
        else:
            st.warning("請輸入要檢測的訊息")

if __name__ == "__main__":
    main()