import os
import sys
import json
import pytest

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.audit_brain_accuracy import run_brain_accuracy_audit

def test_brain_accuracy_audit_execution():
    """
    Ticket 1 TDD 測試:
    - 驗證執行後產出之 data/brain_accuracy_report.json 包含 ic_3d, ic_5d, ic_10d 與 directional_precision_70 鍵值，且皆為有效浮點數。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    report_path = os.path.join(project_root, "data", "brain_accuracy_report.json")
    
    # 1. 移除舊的審計報告
    if os.path.exists(report_path):
        try:
            os.remove(report_path)
        except PermissionError:
            pass
            
    # 2. 執行審計指令
    report = run_brain_accuracy_audit()
    
    # 3. 驗證報告檔案存在
    assert os.path.exists(report_path), "❌ brain_accuracy_report.json 未成功產出！"
    
    # 4. 驗證欄位完整度
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert "ic_3d" in data
    assert "ic_5d" in data
    assert "ic_10d" in data
    assert "directional_precision_70" in data
    
    # 5. 驗證為有效浮點數或整數
    assert isinstance(data["ic_3d"], (int, float))
    assert isinstance(data["ic_5d"], (int, float))
    assert isinstance(data["ic_10d"], (int, float))
    assert isinstance(data["directional_precision_70"], (int, float))
