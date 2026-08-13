import pytest
import pandas as pd
from src.simulate_agents import (
    AgentConfig, AgentState, EntrySignalEvaluator, StrategyPolicyEngine
)

def test_entry_signal_evaluator_tactics_and_fallback():
    cfg_momentum = AgentConfig("A4", "A4_MOMENTUM", "MOMENTUM", 0.60, 0.40, 2.5, 0.6)
    
    # 案例 A：MOMENTUM 未帶量突破，應否決進場
    row_no_breakout = pd.Series({
        'AI_Probability': 0.80, 'Close': 100.0, 'MA_20': 105.0, 'Volume': 500, 'Volume_MA5': 1000
    })
    can_entry, reason = EntrySignalEvaluator.evaluate_entry(row_no_breakout, cfg_momentum)
    assert can_entry is False
    assert reason == "REJECT_MOMENTUM_NO_BREAKOUT"

    # 案例 B：特徵欄位缺失，自動 Graceful Fallback 至純勝率進場
    row_dummy = pd.Series({'AI_Probability': 0.75, 'Close': 100.0})
    can_entry_fb, reason_fb = EntrySignalEvaluator.evaluate_entry(row_dummy, cfg_momentum)
    assert can_entry_fb is True
    assert reason_fb == "ENTRY_PURE_P_CONJ_FALLBACK"

def test_v52_unbound_overheat_and_p_hold_buffer():
    cfg_a1 = AgentConfig("A1", "A1_V_REVERSAL", "V_REVERSAL", 0.60, 0.30, 3.0, 0.6)
    cfg_a9 = AgentConfig("A9", "A9_TOP_DEFENSE", "TOP_DEFENSE", 0.85, 0.55, 2.0, 1.0)
    
    state = AgentState()
    state.shares = 1000
    state.weighted_avg_cost = 100.0

    # 驗證 1：Z-Score = 2.4 時，A1 (Z_bias=3.0) 絕不誤殺
    row_high_z = pd.Series({'AI_Probability': 0.60, 'Z_Score_BIAS': 2.4, 'Close': 105.0})
    dec_a1 = StrategyPolicyEngine.evaluate_v52(row_high_z, state, cfg_a1)
    assert dec_a1["action"] == "HOLD_BUY"

    # 驗證 2：A9 於勝率跌至 0.52 (> 0.50) 時，受 5% 緩衝帶保護不平倉
    row_buffer = pd.Series({'AI_Probability': 0.52, 'Z_Score_BIAS': 1.0, 'Close': 105.0})
    dec_a9_hold = StrategyPolicyEngine.evaluate_v52(row_buffer, state, cfg_a9)
    assert dec_a9_hold["action"] == "HOLD_BUY"

    # 驗證 3：A9 於勝率跌破 0.49 (< 0.50) 時，精確觸發 EXIT_WEAK_P_HOLD_BUFFER
    row_exit = pd.Series({'AI_Probability': 0.49, 'Z_Score_BIAS': 1.0, 'Close': 105.0})
    dec_a9_exit = StrategyPolicyEngine.evaluate_v52(row_exit, state, cfg_a9)
    assert dec_a9_exit["action"] == "SELL"
    assert dec_a9_exit["reason"] == "EXIT_WEAK_P_HOLD_BUFFER"
