import os
import sys
import json
import shutil
import pytest

# 將 src 加入 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_model import execute_v34_closed_loop_rolling_train

def test_weekly_expanding_r2r_and_gsi_evolution():
    """
    Ticket 3 TDD 測試:
    - 驗證重訓後成功產出 data/isd_predict_report.json 與 data/gsi_baseline.npz。
    - 驗證 data/models/ 目錄成功寫入每週歸檔 checkpoint 檔案（如 model_2379_w01_conj.json）。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    isd_path = os.path.join(project_root, "data", "isd_predict_report.json")
    gsi_path = os.path.join(project_root, "data", "gsi_baseline.npz")
    models_dir = os.path.join(project_root, "data", "models")
    
    # 1. 清理舊產出以確保測試正確性 (Windows 下加上 PermissionError 容錯機制)
    for path in [isd_path, gsi_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass
                
    if os.path.exists(models_dir):
        try:
            shutil.rmtree(models_dir)
        except PermissionError:
            pass
        
    # 2. 執行 R2R 滾動重訓管線
    execute_v34_closed_loop_rolling_train()
    
    # 3. 驗證 gsi_baseline.npz 成功產出
    assert os.path.exists(gsi_path), "❌ gsi_baseline.npz 未成功產出！"
    
    # 4. 驗證 isd_predict_report.json 成功產出
    assert os.path.exists(isd_path), "❌ isd_predict_report.json 未成功產出！"
    with open(isd_path, "r", encoding="utf-8") as f:
        isd_data = json.load(f)
    assert "isdTriggered" in isd_data
    assert "gsiValue" in isd_data["riskMetrics"]
    
    # 5. 驗證 data/models/ 目錄成功寫入歸檔 Checkpoint
    assert os.path.exists(models_dir), "❌ models 備份目錄未建立！"
    archive_files = os.listdir(models_dir)
    assert len(archive_files) > 0, "❌ models 目錄下無任何週歸檔備份檔案！"
    assert any("model_2379_w" in name and name.endswith(".json") for name in archive_files), "❌ 未能正確格式化輸出每週 checkpoint (model_2379_wXX_conj.json)！"
