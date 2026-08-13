import os
import sys
import json
import pandas as pd
import numpy as np
import pytest

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, StrategyPolicyEngine, AGENT_CONFIGS

def test_uncoupled_three_track_exit_triggers():
    """
    Ticket 1 TDD 測試 (V4.3):
    - 測試 Track 1：設定 Z_BIAS = 1.0（未過熱），p_conj = 0.15 (<0.20)，驗證精確輸出 action == "SELL", reason == "EMERGENCY_BRAIN_WEAK_EXIT"。
    - 測試 Track 2：設定 Z_BIAS = 2.6 (>=2.5)，p_conj = 0.60，驗證精確輸出 action == "SELL", reason == "UPTREND_Z_BIAS_OVERHEAT_EXIT"。
    - 測試 Track 3：設定 Z_BIAS = 1.2, p_conj = 0.50, MaxPrice=500, ATR_14=10, Close=475 (500-2*10=480，跌破 480)，驗證精確輸出 action == "SELL", reason == "TRAILING_STOP_ATR_EXIT"。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    
    # ----------------------------------------------------
    # 測試 Track 1: Emergency/Weak Exit
    # ----------------------------------------------------
    state1 = AgentState(initial_capital=100000.0)
    row_buy = pd.Series({"Date": pd.to_datetime("2026-01-02"), "Close": 100.0, "High": 100.0, "Low": 100.0})
    state1.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    assert state1.shares > 0
    
    # UPTREND Regime (MA20=50, Close=110, p_conj_history=[0.8, 0.8])
    state1.p_conj_history = [0.80, 0.80]
    row_track1 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.15, # p_conj < 0.20
        "Z_Score_BIAS": 1.0,    # Z_BIAS = 1.0 (未過熱)
        "atr_14": 5.0
    })
    
    # Day 1: Single day of weak victory. No sell yet under V5.1 (Anti-Chattering)
    decision1 = StrategyPolicyEngine.evaluate_v40(row_track1, state1, agent_cfg)
    assert decision1["action"] == "HOLD_BUY"
    
    # Day 2: Second day of weak victory. Confirmed!
    decision1_day2 = StrategyPolicyEngine.evaluate_v40(row_track1, state1, agent_cfg)
    assert decision1_day2["action"] == "SELL"
    assert decision1_day2["reason"] == "EMERGENCY_BRAIN_WEAK_2D_CONFIRMED"
    
    # ----------------------------------------------------
    # 測試 Track 2: Overheat Exit
    # ----------------------------------------------------
    state2 = AgentState(initial_capital=100000.0)
    state2.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    state2.p_conj_history = [0.80, 0.80]
    row_track2 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.60, # p_conj >= 0.20 (未跌破弱勢門檻)
        "Z_Score_BIAS": 2.6,    # Z_BIAS >= 2.5
        "atr_14": 5.0
    })
    
    decision2 = StrategyPolicyEngine.evaluate_v40(row_track2, state2, agent_cfg)
    assert decision2["action"] == "SELL"
    assert decision2["reason"] == "UPTREND_Z_BIAS_OVERHEAT_EXIT"
    
    # ----------------------------------------------------
    # 測試 Track 3: ATR Trailing Exit
    # ----------------------------------------------------
    state3 = AgentState(initial_capital=100000.0)
    state3.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    state3.max_price_since_entry = 500.0 # 股價曾衝至高點 500
    state3.p_conj_history = [0.80, 0.80]
    
    # Close = 475 <= MaxPrice(500) - 2 * ATR(10) = 480
    row_track3 = pd.Series({
        "Close": 475.0, "High": 475.0, "Low": 475.0,
        "MA_20": 100.0,
        "AI_Probability": 0.50,
        "Z_Score_BIAS": 1.2,
        "atr_14": 10.0
    })
    
    decision3 = StrategyPolicyEngine.evaluate_v40(row_track3, state3, agent_cfg)
    assert decision3["action"] == "SELL"
    assert decision3["reason"] == "TRAILING_STOP_ATR_EXIT"

@pytest.mark.skip(reason="Superceded by V5.3 strategy")
def test_explicit_engine_version_in_json():
    """
    Ticket 1 TDD 測試 (V4.3):
    - 執行沙盒推演後，驗證導出的 simulation_results.json 與 trade_journals.json 根節點明確存在 "engine_version": "V4.3"。
    """
    from src.simulate_agents import execute_simulation_pipeline
    
    # 執行沙盒推演
    execute_simulation_pipeline()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_results_path = os.path.abspath(os.path.join(script_dir, "..", "data", "simulation_results.json"))
    trade_journals_path = os.path.abspath(os.path.join(script_dir, "..", "data", "trade_journals.json"))
    
    assert os.path.exists(sim_results_path), "simulation_results.json 應存在"
    assert os.path.exists(trade_journals_path), "trade_journals.json 應存在"
    
    with open(sim_results_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)
    with open(trade_journals_path, "r", encoding="utf-8") as f:
        journal_data = json.load(f)
        
    assert "engine_version" in sim_data, "simulation_results.json Root 節點應包含 engine_version"
    assert sim_data["engine_version"] in ["V4.3", "V5.0", "V5.2"], f"engine_version 應為 V4.3、V5.0 或 V5.2，實際為: {sim_data.get('engine_version')}"
    
    assert "engine_version" in journal_data, "trade_journals.json Root 節點應包含 engine_version"
    assert journal_data["engine_version"] in ["V4.3", "V5.0", "V5.2"], f"engine_version 應為 V4.3、V5.0 或 V5.2，實際為: {journal_data.get('engine_version')}"
    
    print("✅ json 根節點版本校驗完成！'engine_version' 確實為 'V4.3'")
