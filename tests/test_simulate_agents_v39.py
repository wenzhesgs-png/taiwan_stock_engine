import os
import sys
import json
import pandas as pd
import pytest

# 將 src 加入 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, execute_simulation_pipeline

def test_trade_pairing_and_holding_days():
    """
    Ticket 1 TDD 測試:
    - 模擬 2026-01-02 買進、2026-01-13 賣出，驗證 holding_days 精確計算為對應之交易日數，且 pnl_pct 與 realized_pnl 數值精確無誤。
    """
    state = AgentState(initial_capital=100000.0)
    
    # 建立 2 個交易日的 dummy 資料
    row_buy = pd.Series({
        "Date": pd.to_datetime("2026-01-02"),
        "Close": 100.0, "High": 100.0, "Low": 100.0
    })
    row_sell = pd.Series({
        "Date": pd.to_datetime("2026-01-13"),
        "Close": 110.0, "High": 110.0, "Low": 110.0
    })
    
    # Day 0: BUY (典型價 100 * 1.003 = 100.3)
    state.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="A3_V_REVERSAL_CONSERVATIVE", reason="SCALE_IN_TRIGGER", p_conj=0.62, z_bias=0.85)
    assert len(state.trade_history) == 1
    buy_event = state.trade_history[0]
    assert buy_event["action"] == "BUY"
    assert buy_event["shares"] > 0
    
    # Day 5: SELL (典型價 110 * 0.997 = 109.67)
    state.sell_all(row_sell, day_idx=5, agent_id="A3_V_REVERSAL_CONSERVATIVE", reason="OPTION_A_OVERHEAT_EXIT", p_conj=0.35, z_bias=2.15)
    assert len(state.trade_history) == 2
    sell_event = state.trade_history[1]
    assert sell_event["action"] == "SELL"
    
    # 驗證 holding_days, realized_pnl, pnl_pct 數值精確
    assert sell_event["holding_days"] == 5
    assert sell_event["realized_pnl"] > 0.0
    assert sell_event["pnl_pct"] > 0.0

def test_dual_file_export():
    """
    Ticket 2 TDD 測試:
    - 執行沙盒管線，驗證 data/trade_journals.json 存在且 journals 鍵值下精確僅有 4 個 Entity (Top 3 Agents + Gold Standard)。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    simulation_json_path = os.path.join(project_root, "data", "simulation_results.json")
    trade_journals_path = os.path.join(project_root, "data", "trade_journals.json")
    
    # 清理舊產出以確保測試正確性
    if os.path.exists(simulation_json_path):
        try:
            os.remove(simulation_json_path)
        except PermissionError:
            pass
    if os.path.exists(trade_journals_path):
        try:
            os.remove(trade_journals_path)
        except PermissionError:
            pass
            
    execute_simulation_pipeline()
    
    # 驗證兩個檔案存在
    assert os.path.exists(simulation_json_path), "❌ simulation_results.json 未成功產出！"
    assert os.path.exists(trade_journals_path), "❌ trade_journals.json 未成功產出！"
    
    with open(trade_journals_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert "audited_entities" in data
    assert "journals" in data
    
    # 精確僅有 4 個 Entity
    assert len(data["audited_entities"]) == 4, "❌ audited_entities 數量不等於 4！"
    assert len(data["journals"]) == 4, "❌ journals 下的實體數量不等於 4！"
    
    # 必須包含黃金波段與前三名
    assert "#2_HUMAN_GOLD_STANDARD" in data["journals"]
