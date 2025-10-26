"""
使用訓練好的模型進行垃圾郵件預測
"""
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def load_model_and_vectorizer():
    """載入模型和向量器."""
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'spam_classifier.joblib'
    
    # 載入模型
    model = joblib.load(model_path)
    
    # 重新訓練向量器
    # 載入原始數據集以獲取相同的特徵
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
    
    return model, vectorizer

def predict_message(model, vectorizer, message):
    """預測單條訊息是否為垃圾郵件."""
    # 轉換文本為特徵向量
    X = vectorizer.transform([message])
    
    # 預測
    prediction = model.predict(X)[0]
    
    return "垃圾郵件" if prediction == 1 else "正常郵件"

def main():
    # 載入模型和向量器
    print("載入模型...")
    model, vectorizer = load_model_and_vectorizer()
    
    # 測試一些範例
    test_messages = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "Hi honey, how are you? I'll be home late today. Love you!",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "Meeting moved to 3pm tomorrow. Please don't forget to bring your laptop."
    ]
    
    # 進行預測
    print("\n預測結果:")
    print("-" * 50)
    for msg in test_messages:
        result = predict_message(model, vectorizer, msg)
        print(f"\n訊息: {msg[:100]}...")
        print(f"預測: {result}")

if __name__ == "__main__":
    main()