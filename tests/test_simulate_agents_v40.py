import os
import sys
import json
import pandas as pd
import numpy as np
import pytest

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import AgentState, StrategyPolicyEngine, AGENT_CONFIGS, execute_simulation_pipeline

def test_uptrend_trailing_stop_and_holding_duration():
    """
    Ticket 2 TDD 測試:
    - 模擬 UPTREND 情境：最高價達 500 元，價格微幅震盪至 480 元 (回撤 4%)，驗證 p_conj 降至 0.20 時仍不平倉。
    - 模擬價格進一步回撤至 470 元 (回撤 6%)，驗證精確觸發 action == "SELL", reason == "TRAILING_STOP_5PCT_EXIT"。
    - 驗證 9 大 Agent 回測之平均持股天數 holding_days 突破 5 天以上。
    """
    # 選擇 Agent_1 設定進行測試
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    
    state = AgentState(initial_capital=100000.0)
    
    # 模擬買入，最高價達 500 元
    row_buy = pd.Series({
        "Date": pd.to_datetime("2026-01-02"),
        "Close": 500.0, "High": 500.0, "Low": 500.0
    })
    state.buy_tranche(row_buy, percent=1.0, day_idx=0, agent_id="Agent_1")
    assert state.shares > 0
    assert state.max_price_since_entry == 500.0
    
    # UPTREND 狀態：Close > MA20 且 p_conj_3d_avg >= 0.50
    # 我們讓 MA20 為 400.0 (Close 480 > 400)，且 p_conj 的歷史均值為 0.60
    state.p_conj_history = [0.80, 0.80] # 將前兩日設為高勝率
    
    # 價格震盪至 480 元 (回撤 4% <= 5%)，即便 p_conj 降至 0.20，在 UPTREND 下也不應平倉 (Option A 被暫停)
    row_shake = pd.Series({
        "Date": pd.to_datetime("2026-01-13"),
        "Close": 480.0, "High": 480.0, "Low": 480.0,
        "MA_20": 400.0,
        "AI_Probability": 0.20, # 勝率降低
        "Z_Score_BIAS": -1.0
    })
    
    decision1 = StrategyPolicyEngine.evaluate_v40(row_shake, state, agent_cfg)
    assert decision1["action"] != "SELL", "回撤小於 5% 且處於 UPTREND，不應觸發賣出！"
    
    # 價格進一步回撤至 470 元 (回撤 6% > 5%)，觸發 Trailing Stop
    row_crash = pd.Series({
        "Date": pd.to_datetime("2026-01-14"),
        "Close": 470.0, "High": 470.0, "Low": 470.0,
        "MA_20": 400.0,
        "AI_Probability": 0.20,
        "Z_Score_BIAS": -1.0
    })
    
    # 重置勝率歷史以在測試中維持 UPTREND Regime (避免 2 次 0.20 連續輸入導致 Regime 切換為 Rangebound)
    state.p_conj_history = [0.80, 0.80]
    decision2 = StrategyPolicyEngine.evaluate_v40(row_crash, state, agent_cfg)
    assert decision2["action"] == "SELL"
    assert decision2["reason"] == "TRAILING_STOP_5PCT_EXIT"
    
    # 執行全系統回測，驗證其平均持股天數突破 5 天以上
    trade_journals_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trade_journals.json")
    if os.path.exists(trade_journals_path):
        try:
            os.remove(trade_journals_path)
        except PermissionError:
            pass
            
    execute_simulation_pipeline()
    
    assert os.path.exists(trade_journals_path), "trade_journals.json 應存在"
    with open(trade_journals_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 計算各實體的所有交易之平均持股天數 (只看平倉 SELL 的天數)
    total_days = 0
    total_trades = 0
    for entity, journals in data["journals"].items():
        for j in journals:
            if "holding_days" in j:
                total_days += j["holding_days"]
                total_trades += 1
                
    assert total_trades > 0, "回測應包含交易明細"
    avg_holding = total_days / total_trades
    print(f"📊 [動態抗洗盤實測] 平均持股天數: {avg_holding:.2f} 天")
    assert avg_holding > 5.0, f"平均持股天數為 {avg_holding:.2f} 天，未突破 5 天門檻！"
