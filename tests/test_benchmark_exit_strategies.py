import os
import sys
import numpy as np
import pytest

# 將 project root 加入 sys.path 確保能正確匯入 src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.benchmark_exit_strategies import run_exit_benchmark

def test_ab_benchmark_execution():
    """
    Ticket 2 TDD 測試 (V4.3):
    - 驗證執行 A/B 腳本後能順利取得包含 V3.8、V4.2 與 V4.3 結果之字典物件。
    - 驗證 ROI 與 MDD 皆為有效數值（無 NaN 或 Inf）。
    """
    results = run_exit_benchmark()
    
    assert isinstance(results, dict), "結果應為字典物件"
    assert "V3.9" in results, "應包含 V3.9 測試結果"
    assert "V4.2" in results, "應包含 V4.2 測試結果"
    assert "V4.3" in results, "應包含 V4.3 測試結果"
    
    for version in ["V3.9", "V4.2", "V4.3"]:
        stats = results[version]
        assert "roi_pct" in stats, f"{version} 應包含 roi_pct"
        assert "max_drawdown_pct" in stats, f"{version} 應包含 max_drawdown_pct"
        assert "win_rate" in stats, f"{version} 應包含 win_rate"
        assert "avg_holding" in stats, f"{version} 應包含 avg_holding"
        
        roi = stats["roi_pct"]
        mdd = stats["max_drawdown_pct"]
        
        # 驗證數值有效性 (無 NaN, Inf)
        assert not np.isnan(roi), f"{version} roi_pct 包含 NaN"
        assert not np.isinf(roi), f"{version} roi_pct 包含 Inf"
        assert not np.isnan(mdd), f"{version} max_drawdown_pct 包含 NaN"
        assert not np.isinf(mdd), f"{version} max_drawdown_pct 包含 Inf"
        
        print(f"✅ {version} ROI: {roi:+.2f}%, MDD: -{abs(mdd):.2f}%")
