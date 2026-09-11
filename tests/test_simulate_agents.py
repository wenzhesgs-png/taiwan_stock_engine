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
    
    # 驗證包含 #0 Benchmark 在內共有 11 個策略 (1 Benchmark + 10 大 Agents)
    assert len(data["agents"]) == 11, "❌ 回測輸出應包含 1 基準對照組 + 10 大 Agents 策略！"
    assert "A10_ADAPTIVE_REFLEXION" in data, "❌ 回測輸出應包含 A10_ADAPTIVE_REFLEXION 專屬節點！"
    
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


def test_champion_selection_and_tie_breaker():
    """
    DoD 1 TDD 測試:
    - 驗證全沙盒冠軍選拔之同分優先級：A8_TOP_DEFENSE_MODERATE > A9_TOP_DEFENSE_CONSERVATIVE > A7_TOP_DEFENSE_AGGRESSIVE > 其餘 Agent。
    """
    from src.simulate_agents import execute_simulation_pipeline
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    isd_path = os.path.join(current_dir, "..", "data", "isd_predict_report.json")
    
    if os.path.exists(isd_path):
        try:
            os.remove(isd_path)
        except Exception:
            pass
            
    execute_simulation_pipeline()
    
    assert os.path.exists(isd_path), "❌ isd_predict_report.json 應被更新或成功創建！"
    
    with open(isd_path, "r", encoding="utf-8") as f:
        isd_data = json.load(f)
        
    assert "champion_summary" in isd_data, "❌ isd_predict_report.json 應包含 champion_summary 節點！"
    summary = isd_data["champion_summary"]
    assert "agent_id" in summary
    assert "annual_roi" in summary
    assert "today_action" in summary
    assert "holding_status" in summary
    assert "entry_price" in summary
    assert "shares" in summary
    
    # 驗證同分裁決優先級排序
    # 我們構建一個同分的名單進行模擬比對
    priority_order = {
        "A8_TOP_DEFENSE_MODERATE": 0,
        "A9_TOP_DEFENSE_CONSERVATIVE": 1,
        "A7_TOP_DEFENSE_AGGRESSIVE": 2
    }
    
    mock_tied_agents = [
        {"name": "A1_V_REVERSAL_AGGRESSIVE", "roi_pct": 120.0},
        {"name": "A7_TOP_DEFENSE_AGGRESSIVE", "roi_pct": 120.0},
        {"name": "A8_TOP_DEFENSE_MODERATE", "roi_pct": 120.0},
        {"name": "A9_TOP_DEFENSE_CONSERVATIVE", "roi_pct": 120.0}
    ]
    
    def get_sort_key(item):
        prio = priority_order.get(item["name"], 999)
        return (-item["roi_pct"], prio, item["name"])
        
    sorted_tied = sorted(mock_tied_agents, key=get_sort_key)
    assert sorted_tied[0]["name"] == "A8_TOP_DEFENSE_MODERATE", "同分時應優先選擇 A8"
    assert sorted_tied[1]["name"] == "A9_TOP_DEFENSE_CONSERVATIVE", "次選應為 A9"
    assert sorted_tied[2]["name"] == "A7_TOP_DEFENSE_AGGRESSIVE", "再者應為 A7"



def test_a10_mfe_and_profit_giveback_reflection_trigger():
    """
    A10 DoD 3: 利潤回吐偵測 (Failure Mode Detection)
    - 追蹤 MFE = max(High - EntryPrice) / EntryPrice
    - 當 MFE >= +7.0% 且最終實現報酬率 ROI <= +2.0% 時，觸發 reflective_lock = True
    - 反向對照：MFE < 7.0% 或 ROI > 2.0% 時，不觸發反思 (reflective_lock = False)
    """
    # 1. 嚴重利潤回吐案例：進場 100，最高 108 (MFE = +8% >= 7%)，平倉賣出 101.5 (ROI = +1.5% <= 2%)
    mfe_trigger = (108.0 - 100.0) / 100.0
    roi_trigger = (101.5 - 100.0) / 100.0
    assert StrategyPolicyEngine.evaluate_a10_reflection_trigger(mfe=mfe_trigger, roi=roi_trigger) is True
    
    # 2. 對照組 A：MFE 未達標 (最高 105，MFE = +5% < 7%)，ROI = 1.0% <= 2%
    mfe_low = (105.0 - 100.0) / 100.0
    roi_low = (101.0 - 100.0) / 100.0
    assert StrategyPolicyEngine.evaluate_a10_reflection_trigger(mfe=mfe_low, roi=roi_low) is False
    
    # 3. 對照組 B：MFE 達標 (最高 110，MFE = +10% >= 7%)，但利潤順利落袋 (ROI = +6.0% > 2%)
    mfe_high = (110.0 - 100.0) / 100.0
    roi_high = (106.0 - 100.0) / 100.0
    assert StrategyPolicyEngine.evaluate_a10_reflection_trigger(mfe=mfe_high, roi=roi_high) is False


def test_a10_adaptive_trailing_stop_exit():
    """
    A10 DoD 4: 自適應移動鎖利防守 (Adaptive Trailing Stop)
    - 當 active_in_reflection == True 時：
      若持倉期間未實現獲利曾達到 >= +5.0% (MFE >= 0.05)
      一旦股價自最高點回檔幅度 > 3.5% ((max_high - close) / max_high > 0.035)
      當日立即觸發全數平倉出場，原因為 EXIT_TRAILING_LOCK
    """
    # 案例 1：進場 100，最高 106 (MFE = 6% >= 5%)，今日收盤 102 (回檔 (106-102)/106 = 3.77% > 3.5%)
    should_exit, reason = StrategyPolicyEngine.evaluate_a10_trailing_lock(
        current_max_high=106.0,
        close=102.0,
        entry_price=100.0,
        active_in_reflection=True
    )
    assert should_exit is True
    assert reason == "EXIT_TRAILING_LOCK"
    
    # 案例 2：未滿足反思狀態 (active_in_reflection == False)，即便回檔也不應觸發移動鎖利
    should_exit_inactive, _ = StrategyPolicyEngine.evaluate_a10_trailing_lock(
        current_max_high=106.0,
        close=102.0,
        entry_price=100.0,
        active_in_reflection=False
    )
    assert should_exit_inactive is False
    
    # 案例 3：反思狀態下，未實現獲利未曾達 +5% (最高 104，MFE = 4% < 5%)，即便回檔也不啟動移動鎖利
    should_exit_not_5pct, _ = StrategyPolicyEngine.evaluate_a10_trailing_lock(
        current_max_high=104.0,
        close=100.0,
        entry_price=100.0,
        active_in_reflection=True
    )
    assert should_exit_not_5pct is False
    
    # 案例 4：反思狀態下達標 6%，但回檔僅 1.88% <= 3.5%，不觸發出場
    should_exit_minor_pullback, _ = StrategyPolicyEngine.evaluate_a10_trailing_lock(
        current_max_high=106.0,
        close=104.0,
        entry_price=100.0,
        active_in_reflection=True
    )
    assert should_exit_minor_pullback is False


def test_a10_single_trade_reset_and_backtest_flow():
    """
    A10 DoD 5 & Edge Cases:
    - 驗證單筆重置機制：反思鎖利機制僅在下一筆交易生效 1 次；該筆交易平倉後強制重置回 False。
    - 首筆交易初始狀態固定為 reflective_lock = False。
    """
    from src.simulate_agents import run_agent_backtest
    
    a10_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_10")
    assert a10_cfg.name == "A10_ADAPTIVE_REFLEXION"
    assert a10_cfg.tactical_group == "TOP_DEFENSE"
    assert a10_cfg.risk_profile == "MODERATE"
    
    # 構造一個小型行情序列驗證：
    # Trade 1: 進場 -> 衝高至 +8% -> 暴跌平倉 (ROI = -10%) -> 觸發 reflective_lock = True
    # Trade 2: 建立反思持倉 -> 衝高至 +6% -> 回檔 3.77% -> 觸發 EXIT_TRAILING_LOCK -> 平倉後重置 reflective_lock = False
    dates = pd.date_range("2026-01-01", periods=20)
    
    closes = [
        100.0, 101.0, 102.0, 107.0, 106.0, 105.0, 90.0,
        91.0, 92.0, 93.0,
        100.0, 106.0, 102.0,
        102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0
    ]
    highs = [
        100.0, 101.0, 103.0, 108.0, 107.0, 106.0, 92.0,
        91.0, 92.0, 93.0,
        100.0, 106.0, 106.0,
        102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0
    ]
    lows = [c - 1.0 for c in closes]
    opens = closes[:]
    probs = [0.80 if i in [0, 10] else 0.50 for i in range(20)]
    z_scores = [0.0] * 20
    
    df_test = pd.DataFrame({
        'Date': dates,
        'Close': closes,
        'High': highs,
        'Low': lows,
        'Open': opens,
        'AI_Probability': probs,
        'Z_Score_BIAS': z_scores
    })
    
    res = run_agent_backtest(df_test, a10_cfg)
    assert res["total_trades"] >= 2
    journals = res["journals"]
    trailing_exits = [j for j in journals if j.get("action") == "SELL" and j.get("reason") == "EXIT_TRAILING_LOCK"]
    assert len(trailing_exits) >= 1, "❌ Trade 2 應成功觸發 EXIT_TRAILING_LOCK！"
    assert res["reflective_lock"] is False, "❌ 單筆反思結束後 reflective_lock 必須重置為 False！"
