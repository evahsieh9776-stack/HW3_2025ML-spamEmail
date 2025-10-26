"""
預處理垃圾郵件數據集
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    """載入並初步處理數據集."""
    # 讀取數據
    df = pd.read_csv(file_path)
    
    # 重命名列
    df.columns = ['label', 'message']
    
    # 轉換標籤
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    return df

def preprocess_text(df):
    """文本預處理和特徵提取."""
    # 創建TF-IDF向量器
    vectorizer = TfidfVectorizer(
        max_features=1000,  # 限制特徵數量
        stop_words='english',  # 移除停用詞
        lowercase=True,  # 轉換為小寫
        min_df=5  # 最小文檔頻率
    )
    
    # 轉換文本為TF-IDF特徵
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    return X, y, vectorizer

def save_processed_data(X, y, output_dir):
    """保存處理後的數據."""
    # 創建輸出目錄
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存特徵矩陣
    np.save(output_dir / 'features.npy', X.toarray())
    np.save(output_dir / 'labels.npy', y)
    
    print(f"已保存處理後的數據到: {output_dir}")
    print(f"特徵矩陣形狀: {X.shape}")
    print(f"標籤數量: {len(y)}")

def main():
    # 設置路徑
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'datasets' / 'raw'
    processed_dir = base_dir / 'datasets' / 'processed'
    
    # 載入原始數據
    df = load_dataset(raw_dir / 'sms_spam_no_header.csv')
    print("數據載入完成")
    
    # 預處理文本
    X, y, vectorizer = preprocess_text(df)
    print("特徵提取完成")
    
    # 分割訓練和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 保存處理後的數據
    save_processed_data(X_train, y_train, processed_dir / 'train')
    save_processed_data(X_test, y_test, processed_dir / 'test')

if __name__ == "__main__":
    main()