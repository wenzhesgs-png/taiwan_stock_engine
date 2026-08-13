import os
import sys
import pandas as pd
import numpy as np
import pytest

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, StrategyPolicyEngine, AGENT_CONFIGS

def test_agent_state_max_price_lifecycle():
    """
    Ticket 1 TDD 測試:
    - 驗證買進 100 元時 max_price_since_entry == 100。
    - 驗證賣出結算後 max_price_since_entry == 0.0。
    """
    state = AgentState(initial_capital=100000.0)
    assert state.max_price_since_entry == 0.0
    
    # 買進 100 股，價格 100 元
    state.on_buy_executed(shares=100, exec_price=100.0, date="2026-01-02")
    assert state.shares == 100
    assert state.max_price_since_entry == 100.0
    
    # 再買進 50 股，價格 105 元 (最高價應更新為 105)
    state.on_buy_executed(shares=50, exec_price=105.0, date="2026-01-03")
    assert state.shares == 150
    assert state.max_price_since_entry == 105.0
    
    # 賣出結算
    state.on_sell_executed()
    assert state.shares == 0
    assert state.max_price_since_entry == 0.0

def test_trailing_stop_logic_no_false_positive():
    """
    Ticket 2 TDD 測試:
    - 模擬連續 3 天價格每日上漲 1% (100 -> 101 -> 102)，斷言 action 為 HOLD_BUY，且絕不觸發 SELL。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    state = AgentState(initial_capital=100000.0)
    
    # Day 1: BUY at 100
    state.on_buy_executed(shares=100, exec_price=100.0, date="2026-01-02")
    
    # Day 2: Price rises to 101.0 (UPTREND)
    row_day2 = pd.Series({
        "Close": 101.0, "High": 101.0, "Low": 101.0,
        "MA_20": 90.0,
        "AI_Probability": 0.80, # UPTREND: close > ma20 and p_conj_3d_avg >= 0.10
        "Z_Score_BIAS": -1.0
    })
    decision2 = StrategyPolicyEngine.evaluate_v40(row_day2, state, agent_cfg)
    assert decision2["action"] == "HOLD_BUY"
    assert state.max_price_since_entry == 101.0
    
    # Day 3: Price rises to 102.0
    row_day3 = pd.Series({
        "Close": 102.0, "High": 102.0, "Low": 102.0,
        "MA_20": 90.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0
    })
    decision3 = StrategyPolicyEngine.evaluate_v40(row_day3, state, agent_cfg)
    assert decision3["action"] == "HOLD_BUY"
    assert state.max_price_since_entry == 102.0

def test_trailing_stop_actual_5pct_drop():
    """
    Ticket 2 TDD 測試:
    - 模擬價格從 100 上漲至 110 (刷新 max_price)，隨後拉回至 104 (回撤 5.45%)，驗證精確觸發 action == "SELL", reason == "TRAILING_STOP_5PCT_EXIT"。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    state = AgentState(initial_capital=100000.0)
    
    # BUY at 100
    state.on_buy_executed(shares=100, exec_price=100.0, date="2026-01-02")
    
    # Price rises to 110 (UPTREND)
    row_rise = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 90.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0
    })
    decision_rise = StrategyPolicyEngine.evaluate_v40(row_rise, state, agent_cfg)
    assert decision_rise["action"] == "HOLD_BUY"
    assert state.max_price_since_entry == 110.0
    
    # Pull back to 104 (104 <= 110 * 0.95 = 104.5) => Trigger Trailing Stop
    row_drop = pd.Series({
        "Close": 104.0, "High": 104.0, "Low": 104.0,
        "MA20": 90.0, # Support both MA20 and MA_20 keys
        "MA_20": 90.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0
    })
    state.p_conj_history = [0.80, 0.80] # Keep UPTREND regime
    decision_drop = StrategyPolicyEngine.evaluate_v40(row_drop, state, agent_cfg)
    assert decision_drop["action"] == "SELL"
    assert decision_drop["reason"] == "TRAILING_STOP_5PCT_EXIT"
