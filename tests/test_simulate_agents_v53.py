import pytest
import pandas as pd
from src.simulate_agents import StrategyPolicyEngine

def test_evaluate_v53_risk_profiled_entry():
    # 1. Aggressive 旁路門檻 >= 0.70
    res_agg_buy = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.71,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="AGGRESSIVE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_agg_buy["action"] == "BUY"
    assert res_agg_buy["suggested_position"] == 1.0

    res_agg_hold = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.69,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="AGGRESSIVE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_agg_hold["action"] == "HOLD"
    assert res_agg_hold["suggested_position"] == 0.0

    # 2. Moderate 旁路門檻 >= 0.75
    res_mod_buy = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.76,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_mod_buy["action"] == "BUY"
    assert res_mod_buy["suggested_position"] == 1.0

    res_mod_hold = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.74,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_mod_hold["action"] == "HOLD"
    assert res_mod_hold["suggested_position"] == 0.0

    # 3. Conservative 旁路門檻 >= 0.80
    res_con_buy = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.81,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="CONSERVATIVE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_con_buy["action"] == "BUY"
    assert res_con_buy["suggested_position"] == 1.0

    res_con_hold = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.79,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="CONSERVATIVE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_con_hold["action"] == "HOLD"
    assert res_con_hold["suggested_position"] == 0.0


def test_evaluate_v53_isd_override():
    # ISD 風控一票否決 (即便勝率 0.99 且未持倉)
    res_isd_no_pos = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.99,
        z_bias=1.0,
        current_position=0.0,
        unrealized_pnl_pct=0.0,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=True
    )
    assert res_isd_no_pos["action"] == "EXIT"
    assert res_isd_no_pos["suggested_position"] == 0.0

    # ISD 風控一票否決 (即便勝率 0.99 且已持倉)
    res_isd_has_pos = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.99,
        z_bias=1.0,
        current_position=1.0,
        unrealized_pnl_pct=0.05,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=True
    )
    assert res_isd_has_pos["action"] == "EXIT"
    assert res_isd_has_pos["suggested_position"] == 0.0


def test_evaluate_v53_three_track_exit_and_holding():
    # 軌道 1：硬停損 (浮虧 <= -0.08)
    res_stop_loss = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.90,
        z_bias=1.0,
        current_position=1.0,
        unrealized_pnl_pct=-0.08,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_stop_loss["action"] == "EXIT"
    assert res_stop_loss["suggested_position"] == 0.0

    # 軌道 2：高檔過熱雙重確認 (z_bias >= overheat_z_bias)
    # A. 勝率保持強勢 (p_conj >= 0.60) -> 續抱
    res_overheat_strong = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.60,
        z_bias=2.6,
        current_position=0.8,
        unrealized_pnl_pct=0.10,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_overheat_strong["action"] == "HOLD"
    assert res_overheat_strong["suggested_position"] == 0.8

    # B. 勝率轉弱 (p_conj < 0.60) -> 雙重確認成立，離場
    res_overheat_weak = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.59,
        z_bias=2.5,
        current_position=0.8,
        unrealized_pnl_pct=0.10,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_overheat_weak["action"] == "EXIT"
    assert res_overheat_weak["suggested_position"] == 0.0

    # 軌道 3：趨勢轉弱離場 (z_bias < overheat_z_bias)
    # 勝率持續低迷跌破轉弱門檻 (p_conj < 0.40) -> 離場
    res_trend_weak = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.39,
        z_bias=1.0,
        current_position=1.0,
        unrealized_pnl_pct=0.02,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_trend_weak["action"] == "EXIT"
    assert res_trend_weak["suggested_position"] == 0.0

    # 常態續抱區間 (Normal Holding Zone)
    # 勝率維持安全水準 (p_conj >= 0.40)
    res_normal_hold = StrategyPolicyEngine.evaluate_v53(
        p_conj=0.40,
        z_bias=1.0,
        current_position=0.7,
        unrealized_pnl_pct=0.03,
        agent_risk_profile="MODERATE",
        overheat_z_bias=2.5,
        isd_triggered=False
    )
    assert res_normal_hold["action"] == "HOLD"
    assert res_normal_hold["suggested_position"] == 0.7
