import os
import sys
import pandas as pd
import numpy as np
import pytest

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, StrategyPolicyEngine, AGENT_CONFIGS

def test_dual_track_atr_exit_logic():
    """
    Ticket 2 TDD 測試 (V4.2):
    - 測試 Track 1：設定 Z_BIAS = 2.6, p_conj = 0.25，驗證精確輸出 action == "SELL", reason == "UPTREND_Z_BIAS_OVERHEAT_EXIT"。
    - 測試 Track 2：設定 MaxPrice=500, ATR_14=10, Close=475，驗證精確輸出 action == "SELL", reason == "TRAILING_STOP_ATR_EXIT"。
    - 測試波幅呼吸：設定 MaxPrice=500, ATR_14=10, Close=485，驗證輸出 action == "HOLD_BUY"。
    - 測試冷啟動 Fallback 降級：若 atr_14 為 NaN, Close=480, MaxPrice=500 (回撤 4% <= 3% fallback 門檻)，驗證仍為 "HOLD_BUY"；
      若 Close=460 (回撤 8% > 3% fallback 門檻)，驗證精確輸出 action == "SELL", reason == "TRAILING_STOP_FIXED_3PCT_FALLBACK"。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    
    # ----------------------------------------------------
    # 測試 Track 1: 高檔過熱噴發預警
    # ----------------------------------------------------
    state1 = AgentState(initial_capital=100000.0)
    # 模擬已買入持倉
    row_buy = pd.Series({"Date": pd.to_datetime("2026-01-02"), "Close": 100.0, "High": 100.0, "Low": 100.0})
    state1.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    assert state1.shares > 0
    
    # UPTREND Regime (MA20=50, Close=110, p_conj_history=[0.8, 0.8])
    state1.p_conj_history = [0.80, 0.80]
    row_track1 = pd.Series({
        "Close": 110.0, "High": 110.0, "Low": 110.0,
        "MA_20": 50.0,
        "AI_Probability": 0.25, # p_conj < 0.30
        "Z_Score_BIAS": 2.6,    # Z_BIAS >= 2.5
        "atr_14": 5.0
    })
    
    decision1 = StrategyPolicyEngine.evaluate_v40(row_track1, state1, agent_cfg)
    assert decision1["action"] == "SELL"
    assert decision1["reason"] == "UPTREND_Z_BIAS_OVERHEAT_EXIT"
    
    # ----------------------------------------------------
    # 測試 Track 2: 動態波幅停利軌道 (觸發平倉)
    # ----------------------------------------------------
    state2 = AgentState(initial_capital=100000.0)
    state2.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    state2.max_price_since_entry = 500.0 # 股價曾衝至高點 500
    state2.p_conj_history = [0.80, 0.80]
    
    # Close = 475 <= MaxPrice(500) - 2 * ATR(10) = 480
    row_track2 = pd.Series({
        "Close": 475.0, "High": 475.0, "Low": 475.0,
        "MA_20": 100.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0,
        "atr_14": 10.0
    })
    
    decision2 = StrategyPolicyEngine.evaluate_v40(row_track2, state2, agent_cfg)
    assert decision2["action"] == "SELL"
    assert decision2["reason"] == "TRAILING_STOP_ATR_EXIT"
    
    # ----------------------------------------------------
    # 測試波幅呼吸 (不觸發平倉)
    # ----------------------------------------------------
    state3 = AgentState(initial_capital=100000.0)
    state3.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    state3.max_price_since_entry = 500.0
    state3.p_conj_history = [0.80, 0.80]
    
    # Close = 485 > MaxPrice(500) - 2 * ATR(10) = 480
    row_breath = pd.Series({
        "Close": 485.0, "High": 485.0, "Low": 485.0,
        "MA_20": 100.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0,
        "atr_14": 10.0
    })
    
    decision3 = StrategyPolicyEngine.evaluate_v40(row_breath, state3, agent_cfg)
    assert decision3["action"] == "HOLD_BUY"
    
    # ----------------------------------------------------
    # 測試冷啟動 Fallback 降級 (atr_14 為 NaN)
    # ----------------------------------------------------
    state4 = AgentState(initial_capital=100000.0)
    state4.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    state4.max_price_since_entry = 500.0
    state4.p_conj_history = [0.80, 0.80]
    
    # a. 回撤 4% (Close=480)，不應觸發 3% 停損
    row_fallback_hold = pd.Series({
        "Close": 480.0, "High": 480.0, "Low": 480.0,
        "MA_20": 100.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0,
        "atr_14": np.nan
    })
    decision4a = StrategyPolicyEngine.evaluate_v40(row_fallback_hold, state4, agent_cfg)
    assert decision4a["action"] == "HOLD_BUY"
    
    # b. 回撤 8% (Close=460 <= 500 * 0.97 = 485)，應精確觸發 3% 停損
    row_fallback_sell = pd.Series({
        "Close": 460.0, "High": 460.0, "Low": 460.0,
        "MA_20": 100.0,
        "AI_Probability": 0.80,
        "Z_Score_BIAS": -1.0,
        "atr_14": np.nan
    })
    state4.p_conj_history = [0.80, 0.80]
    decision4b = StrategyPolicyEngine.evaluate_v40(row_fallback_sell, state4, agent_cfg)
    assert decision4b["action"] == "SELL"
    assert decision4b["reason"] == "TRAILING_STOP_FIXED_3PCT_FALLBACK"
