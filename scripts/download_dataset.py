"""
下載並整合 Packt 的垃圾郵件分類範例資料集
"""
import os
import shutil
import requests
import zipfile
from pathlib import Path

def download_repository(repo_url, target_dir):
    """Download GitHub repository as zip."""
    # 將 GitHub 倉庫 URL 轉換為 zip 下載 URL
    zip_url = repo_url.rstrip('.git') + '/archive/refs/heads/master.zip'
    zip_path = Path(target_dir) / 'repo.zip'
    
    print(f"正在從 {zip_url} 下載範例...")
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    
    # 保存 zip 檔案
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    # 解壓縮
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    print("下載完成！")

def setup_dataset_directory(base_dir):
    """設置資料集目錄結構."""
    datasets_dir = Path(base_dir) / 'datasets'
    raw_dir = datasets_dir / 'raw'
    processed_dir = datasets_dir / 'processed'
    
    # 建立必要的目錄
    for dir_path in [datasets_dir, raw_dir, processed_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    return raw_dir, processed_dir

def copy_relevant_files(repo_dir, raw_dir):
    """複製相關的資料集檔案."""
    # 先找到解壓縮後的目錄名稱
    repo_contents = list(Path(repo_dir).glob('*'))
    if not repo_contents:
        raise RuntimeError("找不到解壓縮後的目錄")
    
    extracted_dir = repo_contents[0]  # 應該只有一個目錄
    
    # 遞迴搜尋所有CSV檔案
    print(f"搜尋所有資料集於: {extracted_dir}")
    
    # 複製所有相關檔案
    found_files = False
    for file in extracted_dir.rglob('*.csv'):
        if 'spam' in file.name.lower() or 'email' in file.name.lower():
            shutil.copy2(file, raw_dir)
            print(f"已複製: {file.name}")
            found_files = True
    
    if not found_files:
        print("警告: 未找到任何相關的CSV檔案")

def main():
    # 設定參數
    REPO_URL = "https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEMP_DIR = Path(BASE_DIR) / 'temp_repo'

    try:
        # 設置目錄
        raw_dir, processed_dir = setup_dataset_directory(BASE_DIR)
        
        # 確保臨時目錄存在
        TEMP_DIR.mkdir(exist_ok=True, parents=True)
        
        # 下載範例
        download_repository(REPO_URL, TEMP_DIR)
        
        # 複製相關檔案
        copy_relevant_files(TEMP_DIR, raw_dir)
        
        print(f"\n資料集已保存到: {raw_dir}")
        print(f"處理後的資料將保存到: {processed_dir}")
        
    finally:
        # 清理臨時檔案
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print("\n清理完成！")

if __name__ == "__main__":
    main()