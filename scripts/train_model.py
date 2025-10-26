"""
訓練垃圾郵件分類模型
"""
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(data_dir):
    """載入處理後的數據."""
    X = np.load(data_dir / 'features.npy')
    y = np.load(data_dir / 'labels.npy')
    return X, y

def train_model(X_train, y_train):
    """訓練SVM分類器."""
    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """評估模型性能."""
    # 預測
    y_pred = model.predict(X_test)
    
    # 輸出分類報告
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '垃圾']))
    
    # 輸出混淆矩陣
    print("\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, model_dir):
    """保存訓練好的模型."""
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / 'spam_classifier.joblib'
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")

def main():
    # 設置路徑
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'datasets' / 'processed'
    model_dir = base_dir / 'models'
    
    # 載入數據
    print("載入訓練數據...")
    X_train, y_train = load_data(processed_dir / 'train')
    X_test, y_test = load_data(processed_dir / 'test')
    
    # 訓練模型
    print("開始訓練模型...")
    model = train_model(X_train, y_train)
    
    # 評估模型
    print("評估模型性能...")
    evaluate_model(model, X_test, y_test)
    
    # 保存模型
    save_model(model, model_dir)

if __name__ == "__main__":
    main()