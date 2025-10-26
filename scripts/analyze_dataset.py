"""
分析垃圾郵件資料集的內容和格式
"""
import pandas as pd
from pathlib import Path

def analyze_dataset(file_path):
    """分析單個數據集文件."""
    print(f"\n分析文件: {file_path.name}")
    print("-" * 50)
    
    # 讀取數據
    df = pd.read_csv(file_path)
    
    # 基本信息
    print(f"資料集大小: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("\n欄位資訊:")
    print(df.info())
    
    # 顯示前幾行
    print("\n前5行資料:")
    print(df.head())
    
    # 如果有標籤列，顯示分類統計
    if 'label' in df.columns:
        print("\n標籤分布:")
        print(df['label'].value_counts())
    elif 'spam' in df.columns:
        print("\n標籤分布:")
        print(df['spam'].value_counts())

def main():
    # 設置數據集目錄
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'datasets' / 'raw'
    
    # 分析所有CSV文件
    for file_path in raw_dir.glob('*.csv'):
        analyze_dataset(file_path)

if __name__ == "__main__":
    main()