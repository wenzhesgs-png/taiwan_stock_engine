import os
import sys
import pandas as pd
import numpy as np
import pytest

# 將 project root 加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, StrategyPolicyEngine, AGENT_CONFIGS

def test_anti_chattering_and_cooldown():
    """
    Ticket 1 TDD 測試 (V5.1):
    - 測試 Anti-Chattering：設定 p_conj 第一天為 0.15，驗證不平倉 (action == "HOLD_BUY")；第二天持續為 0.15，驗證精確觸發平倉。
    - 測試 T+2 Cooldown：平倉後次日傳入 p_conj = 0.95 (高勝率)，驗證因冷卻期未過，硬性輸出 action == "HOLD", reason == "COOLDOWN_ACTIVE"。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    
    state = AgentState(initial_capital=100000.0)
    # 模擬持倉
    row_buy = pd.Series({"Date": pd.to_datetime("2026-01-02"), "Close": 100.0, "High": 100.0, "Low": 100.0})
    state.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    assert state.shares > 0
    
    # UPTREND Regime (MA20=50, Close=110, p_conj_history=[0.8, 0.8])
    state.p_conj_history = [0.80, 0.80]
    
    # Day 1: p_conj = 0.15 (<0.20), but streak = 1 (not confirmed yet)
    row_day1 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.15,
        "Z_Score_BIAS": 1.0,
        "atr_14": 5.0
    })
    decision1 = StrategyPolicyEngine.evaluate_v40(row_day1, state, agent_cfg)
    assert decision1["action"] == "HOLD_BUY"
    assert state.low_p_conj_streak == 1
    
    # Day 2: p_conj = 0.15 (<0.20) again, streak = 2 (confirmed!)
    row_day2 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.15,
        "Z_Score_BIAS": 1.0,
        "atr_14": 5.0
    })
    decision2 = StrategyPolicyEngine.evaluate_v40(row_day2, state, agent_cfg)
    assert decision2["action"] == "SELL"
    assert decision2["reason"] == "EMERGENCY_BRAIN_WEAK_2D_CONFIRMED"
    
    # 執行平倉 (模擬平倉冷卻期計數被置為 2)
    state.sell_all(row_day2, day_idx=2, agent_id=agent_cfg.name, reason=decision2["reason"])
    assert state.shares == 0
    assert state.cooldown_counter == 2
    
    # Day 3: Next day, high p_conj = 0.95. Verify cooldown blocks entry!
    row_day3 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.95,
        "Z_Score_BIAS": 1.0,
        "atr_14": 5.0
    })
    decision3 = StrategyPolicyEngine.evaluate_v40(row_day3, state, agent_cfg)
    assert decision3["action"] == "HOLD"
    assert decision3["reason"] == "COOLDOWN_ACTIVE"
    # 驗證冷卻計數遞減為 1
    assert state.cooldown_counter == 1

def test_agent_personality_unbinding():
    """
    Ticket 2 TDD 測試 (V5.1):
    - 當 p_conj = 0.70 時，驗證保守型 (A3) 輸出 HOLD (未達 0.85)，而積極型 (A1) 精確發動建倉。
    """
    cfg_a1 = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1") # Aggressive (p_entry=0.60)
    cfg_a3 = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_3") # Conservative (p_entry=0.85)
    
    # 保守型 (A3)
    state_a3 = AgentState(initial_capital=100000.0)
    row = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.70, # < 0.85
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_a3 = StrategyPolicyEngine.evaluate_v40(row, state_a3, cfg_a3)
    assert decision_a3["action"] == "HOLD" # 未達門檻，觀望
    
    # 積極型 (A1)
    state_a1 = AgentState(initial_capital=100000.0)
    decision_a1 = StrategyPolicyEngine.evaluate_v40(row, state_a1, cfg_a1)
    assert decision_a1["action"] == "BUY" # 超越門檻 0.60，加碼/建倉！
