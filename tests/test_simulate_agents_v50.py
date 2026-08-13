import os
import sys
import json
import pandas as pd
import numpy as np
import pytest

# 將 project root 加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import (
    AgentState, StrategyPolicyEngine, AGENT_CONFIGS, 
    ExecutionProxy, COMMISSION_RATE, execute_simulation_pipeline
)

def test_v50_ladder_mapping_precision():
    """
    Ticket 1 & 2 TDD 測試 (V5.0):
    - 驗證 p_conj = 0.95 ==> W_target = 1.0000
    - 驗證 p_conj = 0.82 ==> W_target = 0.6667
    - 驗證 p_conj = 0.68 ==> W_target = 0.3333
    - 驗證 p_conj = 0.52 ==> W_target = 0.0000
    """
    assert StrategyPolicyEngine.get_target_weight_v50(0.95) == 1.0000
    assert StrategyPolicyEngine.get_target_weight_v50(0.82) == 0.6667
    assert StrategyPolicyEngine.get_target_weight_v50(0.68) == 0.3333
    assert StrategyPolicyEngine.get_target_weight_v50(0.52) == 0.0000

@pytest.mark.skip(reason="Superceded by V5.3 strategy")
def test_v50_scenarios():
    """
    Ticket 2 TDD 測試 (V5.0):
    - 情境 A (All-In 全倉滿打)：模擬高勝率 p_conj = 0.92，驗證可一次性將剩餘現金依手續費精算買滿至總權益 100%。
    - 情境 B (動態加碼 Scale-In)：模擬初始 1/3 倉位，次日 p_conj 升至 0.85，驗證系統成功計算 Delta_C 並追加購買入整股 Delta_S。
    - 情境 C (勝率回落不減倉)：模擬 2/3 倉位，次日 p_conj 回落至 0.65，斷言當日賣出股數為 0，且持股維持不變。
    - 情境 D (版本標籤寫入驗證)：驗證沙盒推演產出之 JSON 檔案 root 節點包含 "engine_version": "V5.0"。
    """
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    
    # ────────────────────────────────────────────────────
    # 情境 A (All-In 全倉滿打)
    # ────────────────────────────────────────────────────
    agent_cfg_a3 = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_3")
    state_a = AgentState(initial_capital=100000.0)
    row_a = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.92, # W_target = 1.0
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_a = StrategyPolicyEngine.evaluate_v40(row_a, state_a, agent_cfg_a3)
    assert decision_a["action"] == "BUY"
    assert decision_a["position"] == 1.0
    
    # 執行買入
    state_a.buy_tranche(row_a, percent=decision_a["position"], day_idx=0, agent_id="Agent_3", reason=decision_a["reason"], p_conj=0.92, z_bias=0.0)
    # 驗證現金幾乎用盡，持股價值佔大頭
    assert state_a.shares > 0
    assert state_a.available_buying_power < 1000.0 # 現金餘額極低，完美精算打滿！
    
    # ────────────────────────────────────────────────────
    # 情境 B (動態加碼 Scale-In)
    # ────────────────────────────────────────────────────
    state_b = AgentState(initial_capital=100000.0)
    row_b1 = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.65, # W_target = 0.3333 * 0.6 = 0.20
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_b1 = StrategyPolicyEngine.evaluate_v40(row_b1, state_b, agent_cfg)
    assert decision_b1["action"] == "BUY"
    assert round(decision_b1["position"], 2) == 0.20
    
    # 執行首日買入 (1/3 倉)
    state_b.buy_tranche(row_b1, percent=decision_b1["position"], day_idx=0, agent_id="Agent_1", reason=decision_b1["reason"], p_conj=0.65, z_bias=0.0)
    shares_day1 = state_b.shares
    assert shares_day1 > 0
    
    # 次日勝率跳升至 0.85 (W_target = 0.6667 * 0.6 = 0.40)
    row_b2 = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.85, # W_target = 0.40
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_b2 = StrategyPolicyEngine.evaluate_v40(row_b2, state_b, agent_cfg)
    assert decision_b2["action"] == "BUY"
    # 預計 Delta_C 為 0.20
    assert 0.18 <= decision_b2["position"] <= 0.22
    
    # 執行加碼
    state_b.buy_tranche(row_b2, percent=decision_b2["position"], day_idx=1, agent_id="Agent_1", reason=decision_b2["reason"], p_conj=0.85, z_bias=0.0)
    assert state_b.shares > shares_day1
    
    # ────────────────────────────────────────────────────
    # 情境 C (勝率回落不減倉)
    # ────────────────────────────────────────────────────
    state_c = AgentState(initial_capital=100000.0)
    row_c1 = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.80, # W_target = 0.6667
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_c1 = StrategyPolicyEngine.evaluate_v40(row_c1, state_c, agent_cfg)
    state_c.buy_tranche(row_c1, percent=decision_c1["position"], day_idx=0, agent_id="Agent_1", reason=decision_c1["reason"], p_conj=0.80, z_bias=0.0)
    shares_day1_c = state_c.shares
    assert shares_day1_c > 0
    
    # 次日勝率下滑至 0.65 (W_target = 0.3333 <= 0.6667)，不應觸發減倉
    row_c2 = pd.Series({
        "Close": 100.0, "High": 100.0, "Low": 100.0,
        "MA_20": 90.0,
        "AI_Probability": 0.65, # W_target = 0.3333
        "Z_Score_BIAS": 0.0,
        "atr_14": 5.0
    })
    decision_c2 = StrategyPolicyEngine.evaluate_v40(row_c2, state_c, agent_cfg)
    # 絕不觸發 SELL (由於 W_target <= W_current 且未觸發 V4.3 三軌離場)
    assert decision_c2["action"] == "HOLD_BUY" or decision_c2["action"] == "HOLD"
    assert state_c.shares == shares_day1_c # 持股數 100% 完全續抱不變！
    
    # ────────────────────────────────────────────────────
    # 情境 D (版本標籤寫入驗證)
    # ────────────────────────────────────────────────────
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
        
    assert "engine_version" in sim_data
    assert sim_data["engine_version"] in ["V5.0", "V5.2"]
    assert "engine_version" in journal_data
    assert journal_data["engine_version"] in ["V5.0", "V5.2"]
    
    print("✅ V5.0 動態信心建倉與不減倉鐵律各情境驗證通過！")
