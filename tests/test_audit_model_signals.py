import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# 將 src 加入 Python 模組搜尋路徑，確保測試能正常匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.audit_model_signals import (
    evaluate_pivot_signal,
    get_nearest_trading_date,
    run_signal_audit
)

def test_evaluator_tagging_logic():
    """
    Ticket 1: 驗證 evaluate_pivot_signal 轉折點標註判定邏輯
    - BUY 點：傳入 p_conj = 0.65，驗證為 [SIGNAL_OK]；傳入 p_conj = 0.35，驗證為 [MODEL_BLIND]。
    - SELL 點：傳入 z_bias = 2.2，驗證為 [SIGNAL_OK]；傳入 z_bias = 0.5, p_conj = 0.60，驗證為 [MODEL_BLIND]。
    """
    # 1. 測試 BUY 點
    assert evaluate_pivot_signal("BUY", 0.65, -1.5) == "[SIGNAL_OK]"
    assert evaluate_pivot_signal("BUY", 0.35, -1.5) == "[MODEL_BLIND]"
    
    # 2. 測試 SELL 點
    # Z_BIAS 過熱清倉
    assert evaluate_pivot_signal("SELL", 0.70, 2.2) == "[SIGNAL_OK]"
    # p_conj 跌破續抱門檻 (0.40) 清倉
    assert evaluate_pivot_signal("SELL", 0.30, 0.5) == "[SIGNAL_OK]"
    # 無信號清倉 (未過熱，勝率在續抱門檻 0.40 以上)
    assert evaluate_pivot_signal("SELL", 0.60, 0.5) == "[MODEL_BLIND]"

def test_get_nearest_trading_date():
    """
    驗證當目標日期遇到非交易日，自動往前（過去時間）尋找最近的一個交易日。
    """
    available_dates = ["2026-01-02", "2026-01-05", "2026-01-06"] # 1/3, 1/4 為週末
    
    # 遇到 1/4 週末，應該往前取最近交易日 1/2
    assert get_nearest_trading_date("2026-01-04", available_dates) == "2026-01-02"
    
    # 遇到 1/5 交易日，應該取 1/5
    assert get_nearest_trading_date("2026-01-05", available_dates) == "2026-01-05"

def test_audit_report_generation():
    """
    Ticket 1: 驗證 run_signal_audit 成果
    - 驗證 data/audit_signals_report.json 寫入成功。
    - 驗證包含 12 個轉折日完整數據。
    - 驗證欄位完整且不含 NaN 或 Inf。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_json_path = os.path.join(current_dir, "..", "data", "audit_signals_report.json")
    
    # 移除舊結果檔
    if os.path.exists(report_json_path):
        os.remove(report_json_path)
        
    # 執行稽核
    run_signal_audit()
    
    # 驗證輸出
    assert os.path.exists(report_json_path), "❌ audit_signals_report.json 未成功產出！"
    
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert "audit_results" in data
    assert len(data["audit_results"]) == 12, "❌ 稽核結果應包含 12 個轉折日完整數據！"
    
    for r in data["audit_results"]:
        assert "target_date" in r
        assert "actual_trading_date" in r
        assert "action_type" in r
        assert "p_conj" in r
        assert "p_ref" in r
        assert "z_bias" in r
        assert "gsi" in r
        assert "status" in r
        
        # 驗證不能含有 NaN 或 Inf
        for key in ["p_conj", "p_ref", "z_bias", "gsi"]:
            val = r[key]
            assert val is not None
            assert not np.isnan(val), f"欄位 {key} 不能為 NaN！"
            assert not np.isinf(val), f"欄位 {key} 不能為 Inf！"
            
        assert r["status"] in ["[SIGNAL_OK]", "[MODEL_BLIND]"]
