import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# 將 src 加入 Python 模組搜尋路徑，確保測試能正常匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulate_agents import (
    calculate_friction_cost,
    ExecutionProxy,
    StrategyPolicyEngine,
    AgentConfig,
    AGENT_CONFIGS,
    run_benchmark_backtest,
    execute_simulation_pipeline
)

def test_typical_price_slippage():
    """
    Ticket 1: 擬真成交價與 0.3% 滑價懲罰
    - High=110, Low=90, Close=100 (Typical Price = 100)
    - 買入成交價精確等於 100.3
    - 賣出成交價精確等於 99.7
    """
    dummy_row = pd.Series({
        'High': 110.0,
        'Low': 90.0,
        'Close': 100.0
    })
    
    tp = ExecutionProxy.get_typical_price(dummy_row)
    assert tp == 100.0
    
    buy_price = ExecutionProxy.get_buy_execution_price(dummy_row)
    assert buy_price == pytest.approx(100.3)
    
    sell_price = ExecutionProxy.get_sell_execution_price(dummy_row)
    assert sell_price == pytest.approx(99.7)

def test_hard_stop_loss_trigger():
    """
    Ticket 2: 8% 單筆物理硬停損與優先級
    - 進場成本為 500 元，現價跌至 455 元 (PnL % = -9%)
    - 即使 p_conj = 0.95 (高勝率指標)，系統仍精確輸出 action == "SELL", reason == "HARD_STOP_LOSS_8PCT", position == 0.0。
    """
    # 選擇 Agent_7 設定進行測試
    agent_7_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_7")
    
    row_stoploss = pd.Series({
        'AI_Probability': 0.95,  # 高勝率
        'Z_Score_BIAS': -1.8,    # 極低 Z-Score (未過熱)
        'Close': 455.0           # 跌破 8% 限度 (455 / 500 = 0.91 -> -9%)
    })
    
    decision = StrategyPolicyEngine.evaluate(row_stoploss, 0.6, 500.0, agent_7_cfg, isd_triggered=False)
    assert decision["action"] == "SELL"
    assert decision["reason"] == "HARD_STOP_LOSS_8PCT"
    assert decision["position"] == 0.0

def test_holding_continuation():
    """
    Ticket 2: 驗證波段續抱
    - 傳入 current_holding = 0.5, p_conj = 0.50 (> p_hold_threshold = 0.40) 且未過熱
    - 驗證輸出 action == "HOLD_BUY" 且維持當前持倉。
    """
    agent_4_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_4")
    
    row_continue = pd.Series({
        'AI_Probability': 0.50,
        'Z_Score_BIAS': 1.0,
        'Close': 100.0
    })
    
    decision = StrategyPolicyEngine.evaluate(row_continue, 0.5, 95.0, agent_4_cfg, isd_triggered=False)
    assert decision["action"] == "HOLD_BUY"
    assert decision["position"] == 0.5

def test_isd_hard_circuit_breaker_all_agents():
    """
    Ticket 3: 驗證 ISD 全域硬熔斷
    - 當 isdTriggered == True 時，所有 9 大 Agent 的當日決策強制壓制為 action: "HOLD", position: 0.0。
    """
    row_today = pd.Series({
        'AI_Probability': 0.85, # 高勝率
        'Z_Score_BIAS': -1.8,
        'Close': 100.0
    })
    
    for cfg in AGENT_CONFIGS:
        decision = StrategyPolicyEngine.evaluate(row_today, 0.6, 90.0, cfg, isd_triggered=True)
        assert decision["action"] == "HOLD"
        assert decision["position"] == 0.0
        assert decision["reason"] == "ISD_CIRCUIT_BREAKER"

def test_benchmark_alignment():
    """
    Ticket 3: Benchmark 基準對照組測試
    - 傳入 100 天測試數據，驗證產出之 benchmark 鍵值存在，且總 ROI 正確。
    """
    # 建立 100 天虛擬數據 (Close 從 100 一路漲到 120)
    dates = pd.date_range(start="2026-01-01", periods=100)
    closes = np.linspace(100.0, 120.0, 100)
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Close': closes,
        'High': closes,
        'Low': closes,
        'Open': closes
    })
    
    bench_stats = run_benchmark_backtest(dummy_data)
    assert bench_stats["final_capital"] > 0
    assert "roi_pct" in bench_stats
    assert "max_drawdown_pct" in bench_stats
    assert bench_stats["buy_count"] == 1
    assert bench_stats["sell_count"] == 1

def test_full_simulation_pipeline():
    """
    Ticket 3: 驗證整套 execute_simulation_pipeline 的整合回測與導出
    - 驗證 data/simulation_results.json 寫入成功且格式合法。
    - 驗證 #0_BENCHMARK_BUY_HOLD 正確輸出。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_json_path = os.path.join(current_dir, "..", "data", "simulation_results.json")
    
    if os.path.exists(results_json_path):
        os.remove(results_json_path)
        
    execute_simulation_pipeline()
    
    assert os.path.exists(results_json_path), "❌ simulation_results.json 未成功產出！"
    
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert "isdTriggered" in data
    assert "agents" in data
    
    # 驗證包含 #0 Benchmark 在內共有 10 個策略 (1 Benchmark + 9 Agents)
    assert len(data["agents"]) == 10, "❌ 回測輸出應包含 1 基準對照組 + 9 大 Agents 策略！"
    
    benchmark_agent = next(a for a in data["agents"] if a["agent_id"] == "Agent_0")
    assert benchmark_agent["name"] == "#0_BENCHMARK_BUY_HOLD"
    assert benchmark_agent["tactical_group"] == "BENCHMARK"
    assert "roi_pct" in benchmark_agent["backtest"]
    assert "buy_count" in benchmark_agent["backtest"]
    assert "sell_count" in benchmark_agent["backtest"]

def test_tiered_discount_fee_and_monthly_reset():
    """
    Ticket 2 TDD 測試:
    - 測試買進 500,000 元，手續費精確等於 ⌊500000×0.001425×0.2⌋=142 元。
    - 測試跨月（如 01/31 到 02/01）成交時，驗證當月累計金額精確重置，重新享有 2 折優惠。
    """
    from src.simulate_agents import AgentState, SettlementEngine
    
    state = AgentState(initial_capital=1000000.0)
    
    # 測試第一天 01/31 交易 500,000 元
    fee1 = SettlementEngine.calculate_fee(state, 500000.0, "2026-01-31")
    assert fee1 == 142.0
    assert state.monthly_turnover == 500000.0
    assert state.last_trade_date == pd.to_datetime("2026-01-31")
    
    # 測試第二天 01/31 再次交易 600,000 元 (跨越 100 萬門檻)
    # 前 500,000 元在 2 折額度內，後 100,000 元為 6.5 折
    # fee_raw = (500000 * 0.2 + 100000 * 0.65) * 0.001425 = 165000 * 0.001425 = 235.125
    # fee_final = floor(235.125) = 235
    fee2 = SettlementEngine.calculate_fee(state, 600000.0, "2026-01-31")
    assert fee2 == 235.0
    assert state.monthly_turnover == 1100000.0
    
    # 測試第三天 02/01 跨月交易 500,000 元 (應該重置為 0，重新享有 2 折)
    fee3 = SettlementEngine.calculate_fee(state, 500000.0, "2026-02-01")
    assert fee3 == 142.0
    assert state.monthly_turnover == 500000.0
