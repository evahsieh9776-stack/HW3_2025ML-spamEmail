"""
垃圾郵件分類器網頁應用
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go

# 設置頁面配置
st.set_page_config(
    page_title="垃圾郵件分類器",
    page_icon="✉️",
    layout="wide"
)

# 載入模型和向量器
@st.cache_resource
def load_model_and_vectorizer():
    """載入模型和向量器."""
    base_dir = Path(__file__).parent
    model_path = base_dir / 'models' / 'spam_classifier.joblib'
    
    # 載入模型
    model = joblib.load(model_path)
    
    # 重新訓練向量器
    raw_dir = base_dir / 'datasets' / 'raw'
    df = pd.read_csv(raw_dir / 'sms_spam_no_header.csv')
    df.columns = ['label', 'message']
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True,
        min_df=5
    )
    vectorizer.fit(df['message'])
    
    return model, vectorizer, df

def predict_message(model, vectorizer, message):
    """預測單條訊息是否為垃圾郵件."""
    # 轉換文本為特徵向量
    X = vectorizer.transform([message])
    
    # 預測
    prediction = model.predict(X)[0]
    # 獲取決策函數值（置信度）
    confidence = abs(model.decision_function(X)[0])
    
    return prediction, confidence

def create_word_importance_plot(vectorizer, model, message):
    """創建單字重要性的視覺化."""
    # 獲取特徵名稱（單字）
    feature_names = vectorizer.get_feature_names_out()
    
    # 獲取訊息的特徵向量
    X = vectorizer.transform([message])
    
    # 獲取每個特徵的權重
    weights = X.toarray()[0] * model.coef_[0]
    
    # 建立DataFrame
    word_importance = pd.DataFrame({
        '單字': feature_names,
        '重要性': weights
    })
    
    # 只保留有權重的單字
    word_importance = word_importance[word_importance['重要性'] != 0]
    
    # 排序並取前15個最重要的單字
    word_importance = word_importance.nlargest(15, '重要性')
    
    # 創建長條圖
    fig = px.bar(
        word_importance,
        x='重要性',
        y='單字',
        orientation='h',
        title='影響分類結果的關鍵字',
        labels={'重要性': '影響程度', '單字': ''},
    )
    
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(height=400)
    
    return fig

def create_confidence_gauge(confidence):
    """創建置信度儀表板."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': 'lightgray'},
                {'range': [1, 2], 'color': 'gray'},
                {'range': [2, 5], 'color': 'lightblue'}
            ],
        },
        title = {'text': "分類置信度"}
    ))
    
    fig.update_layout(height=250)
    return fig

def create_dataset_stats(df):
    """創建資料集統計視覺化."""
    # 計算標籤分布
    label_counts = df['label'].value_counts()
    
    # 創建圓餅圖
    fig = px.pie(
        values=label_counts.values,
        names=['正常郵件', '垃圾郵件'],
        title='資料集分布',
        color_discrete_sequence=['lightblue', 'gray']
    )
    
    fig.update_layout(height=300)
    return fig

def main():
    # 載入模型和數據
    model, vectorizer, df = load_model_and_vectorizer()
    
    # 頁面標題
    st.title("✉️ 垃圾郵件偵測系統")
    st.write("---")
    
    # 創建兩欄布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 輸入區域
        st.subheader("📝 輸入要檢查的訊息")
        message = st.text_area(
            "在此輸入郵件內容",
            height=200,
            key="message_input"
        )
        
        if st.button("進行分析", type="primary"):
            if message:
                # 進行預測
                prediction, confidence = predict_message(model, vectorizer, message)
                
                # 顯示結果
                st.write("---")
                st.subheader("🔍 分析結果")
                
                # 使用顏色區塊顯示結果
                if prediction == 1:
                    st.error("⚠️ 這可能是垃圾郵件")
                else:
                    st.success("✅ 這可能是正常郵件")
                
                # 顯示置信度儀表板
                st.plotly_chart(create_confidence_gauge(confidence))
                
                # 顯示關鍵字分析
                st.plotly_chart(create_word_importance_plot(vectorizer, model, message))
            else:
                st.warning("請輸入要分析的訊息")
    
    with col2:
        # 顯示資料集統計
        st.subheader("📊 訓練資料統計")
        st.plotly_chart(create_dataset_stats(df))
        
        # 顯示模型資訊
        st.subheader("ℹ️ 模型資訊")
        st.info("""
        - 使用線性支持向量機(SVM)分類器
        - 準確率：98%
        - 訓練資料量：5,573筆
        """)
        
        # 顯示使用說明
        st.subheader("📖 使用說明")
        st.write("""
        1. 在左側輸入想要檢查的郵件內容
        2. 點擊「進行分析」按鈕
        3. 系統會顯示:
           - 分類結果
           - 分類置信度
           - 影響分類的關鍵字
        """)

if __name__ == "__main__":
    main()