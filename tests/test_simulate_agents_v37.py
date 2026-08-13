import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# 將 src 加入 Python 模組搜尋路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulate_agents import (
    calculate_friction_cost,
    ExecutionProxy,
    StrategyPolicyEngine,
    AgentConfig,
    AGENT_CONFIGS,
    run_benchmark_backtest,
    execute_simulation_pipeline,
)

# V3.7 新引入的元件，我們預期在實作前會匯入失敗或未實作
# 為了進行 TDD Step 1 (Red)，我們先嘗試導入。如果尚未實作，測試執行時自然會 Fail
try:
    from src.simulate_agents import (
        BenchmarkRunner,
        AgentState,
        calculate_score_mapping,
    )
except ImportError:
    # 建立 Mock 以免編譯/語法層面直接掛掉，但測試內容會觸發 Failure
    BenchmarkRunner = None
    AgentState = None
    calculate_score_mapping = None


def test_benchmark_runner_execution():
    """
    1. 驗證 3 大對照組實體撮合 (BenchmarkRunner)
    - #0_BENCHMARK_BUY_HOLD: 2026-01-02 買入 100% -> 2026-07-31 賣出
    - #1_HUMAN_REAL_TRADE: 2026-01-26 買 100 股, 2026-03-05 買 55 股, 2026-06-18 賣 155 股
    - #2_HUMAN_GOLD_STANDARD: 6 大黃金波段複利滾動
    """
    assert BenchmarkRunner is not None, "BenchmarkRunner 未實作！"
    
    # 建立包含這 16 個關鍵日期的測試資料
    dates = [
        "2026-01-02", "2026-01-13", "2026-01-26", "2026-03-04", "2026-03-05", 
        "2026-04-22", "2026-05-04", "2026-05-11", "2026-05-15", "2026-06-03", 
        "2026-06-08", "2026-06-18", "2026-06-23", "2026-06-26", "2026-07-09", 
        "2026-07-31"
    ]
    # 為了方便測試，我們隨便給些收盤價，例如穩步上升的價格，或特定價格
    # 讓 TypicalPrice = Close = High = Low
    closes = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0]
    
    df_dummy = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Close": closes,
        "High": closes,
        "Low": closes,
        "Open": closes
    })
    df_dummy["Date_Str"] = df_dummy["Date"].dt.strftime("%Y-%m-%d")
    
    runner = BenchmarkRunner(initial_capital=100000.0)
    results = runner.run_all(df_dummy)
    
    assert "#0_BENCHMARK_BUY_HOLD" in results
    assert "#1_HUMAN_REAL_TRADE" in results
    assert "#2_HUMAN_GOLD_STANDARD" in results
    
    # #0 驗證：買進 Close 是 100.0，賣出 Close 是 250.0
    bh_res = results["#0_BENCHMARK_BUY_HOLD"]
    assert bh_res["roi_pct"] > 0
    
    # #1 驗證
    hr_res = results["#1_HUMAN_REAL_TRADE"]
    assert "roi_pct" in hr_res
    
    # #2 驗證
    gs_res = results["#2_HUMAN_GOLD_STANDARD"]
    assert gs_res["roi_pct"] > bh_res["roi_pct"], "黃金波段複利滾動應該有極高報酬"


def test_multi_tranche_weighted_stop_loss_and_instant_bp():
    """
    2. 驗證動態多階建倉、加權成本、硬停損、即時購買力解凍引擎 (AgentState & StrategyPolicyEngine V3.7)
    """
    assert AgentState is not None, "AgentState 未實作！"
    
    state = AgentState(initial_capital=100000.0)
    
    # 測試多階建倉 (Scale-In) 預算 33% 與加權成本計算
    # 假設第一天 TypicalPrice = 100, 買入一階 (33% 總權益 = 33000)
    # 買入成交價 EP_buy = 100 * 1.003 = 100.3
    # 買入股數 = 33000 / (100.3 * 1.001425) = 328 股
    row_1 = pd.Series({"Close": 100.0, "High": 100.0, "Low": 100.0, "AI_Probability": 0.60, "Z_Score_BIAS": 0.0})
    
    state.buy_tranche(row_1, percent=0.33)
    assert state.tranches == 1
    assert state.shares > 0
    assert state.weighted_avg_cost == pytest.approx(100.3)
    
    # 記錄此時購買力與實體資金
    bp_after_buy1 = state.available_buying_power
    cash_after_buy1 = state.settled_cash
    
    # 第二天 TypicalPrice = 90, 勝率 0.60 再次加碼
    row_2 = pd.Series({"Close": 90.0, "High": 90.0, "Low": 90.0, "AI_Probability": 0.60, "Z_Score_BIAS": 0.0})
    state.buy_tranche(row_2, percent=0.33)
    assert state.tranches == 2
    # 驗證加權成本變低
    assert state.weighted_avg_cost < 100.3
    assert state.weighted_avg_cost > 90.27  # 90 * 1.003
    
    # 測試 8% 加權硬停損
    # 加權成本大約在 95 左右。若價格暴跌至 80，則 (80 - AvgCost)/AvgCost 跌幅必超過 8%
    row_crash = pd.Series({"Close": 80.0, "High": 80.0, "Low": 80.0, "AI_Probability": 0.99, "Z_Score_BIAS": -2.0})
    # 即使勝率極高 (0.99)，因為跌幅超過 8%，也必須強制 SELL
    agent_cfg = next(cfg for cfg in AGENT_CONFIGS if cfg.agent_id == "Agent_1")
    decision = StrategyPolicyEngine.evaluate_v37(row_crash, state, agent_cfg)
    assert decision["action"] == "SELL"
    assert decision["reason"] == "HARD_STOP_LOSS_8PCT"
    
    # 測試即時購買力解凍 (T+0 BP Settlement)
    # 賣出時，BP 應該立刻增加賣出淨收益 (T+0)
    # 而 settled_cash 維持原本，直到 T+2 才解凍
    prev_bp = state.available_buying_power
    prev_cash = state.settled_cash
    
    state.sell_all(row_crash, day_idx=10)
    assert state.shares == 0
    assert state.tranches == 0
    assert state.weighted_avg_cost == 0.0
    
    # 即時購買力立刻大於賣出前
    assert state.available_buying_power > prev_bp
    # 實體資金當天不變
    assert state.settled_cash == prev_cash
    
    # 過了 2 天 (day_idx >= 12)，交割到帳，實體資金解凍
    state.update_settlement(day_idx=12)
    assert state.settled_cash > prev_cash


def test_score_mapping_logic():
    """
    3. 驗證戰力分級內插算式 (Score Mapping)
    - Buy&Hold 為 40 分 Pass
    - HumanReal 為 60 分 Good
    - GoldStandard 為 90 分 Superior
    - 超過 GoldStandard 可以突破 90 分甚至 100+ 分
    """
    assert calculate_score_mapping is not None, "calculate_score_mapping 未實作！"
    
    r_bh = 10.0   # Buy & Hold ROI 10%
    r_hr = 25.0   # Human Real ROI 25%
    r_gs = 60.0   # Gold Standard ROI 60%
    
    # 精確對齊基準點
    assert calculate_score_mapping(r_bh, r_bh, r_hr, r_gs) == pytest.approx(40.0)
    assert calculate_score_mapping(r_hr, r_bh, r_hr, r_gs) == pytest.approx(60.0)
    assert calculate_score_mapping(r_gs, r_bh, r_hr, r_gs) == pytest.approx(90.0)
    
    # 區間內插
    score_mid1 = calculate_score_mapping(17.5, r_bh, r_hr, r_gs) # 介於 bh 與 hr 中點
    assert score_mid1 == pytest.approx(50.0)
    
    score_mid2 = calculate_score_mapping(42.5, r_bh, r_hr, r_gs) # 介於 hr 與 gs 中點
    assert score_mid2 == pytest.approx(75.0)
    
    # 突破上限
    score_high = calculate_score_mapping(80.0, r_bh, r_hr, r_gs)
    assert score_high > 90.0
    
    # 低於下限
    score_low = calculate_score_mapping(0.0, r_bh, r_hr, r_gs)
    assert score_low < 40.0
