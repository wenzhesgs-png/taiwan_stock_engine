import os
import sys
import numpy as np
import pandas as pd
import pytest

# 將 src 加入 Python 模組搜尋路徑，確保測試能正常匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict_today import calculate_gsi, calculate_reliance_index, evaluate_isd_circuit_breaker

def test_calculate_reliance_index():
    """
    驗證 Reliance Index (RI) 完美符合 1.0 - |p_ref - p_conj| 公式。
    """
    # 正常情況
    ri = calculate_reliance_index(0.72, 0.78)
    assert round(ri, 4) == 0.9400
    
    # 歷史趨勢嚴重背離情況
    ri_extreme = calculate_reliance_index(0.10, 0.90)
    assert round(ri_extreme, 4) == 0.2000

def test_calculate_gsi_success_and_fallback():
    """
    1. 驗證 GSI 成功以馬氏距離計算。
    2. 驗證協方差奇異或包含 NaN 時，降級為 Standardized Euclidean (seuclidean) fallback。
    """
    # 正確無損的協常數與特徵空間
    current_features = np.array([1.0, 2.0])
    mean = np.array([1.2, 1.8])
    inv_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    std_vector = np.array([0.5, 0.5])
    
    gsi = calculate_gsi(current_features, mean, inv_cov, std_vector)
    # delta = [-0.2, 0.2]
    # dist_sq = (-0.2)*1*(-0.2) + (0.2)*1*(0.2) = 0.04 + 0.04 = 0.08
    # gsi = sqrt(0.08) = 0.2828427
    assert round(gsi, 4) == round(np.sqrt(0.08), 4)
    
    # 協方差逆矩陣損壞（包含 NaN / Inf）
    inv_cov_bad = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    gsi_fallback = calculate_gsi(current_features, mean, inv_cov_bad, std_vector)
    # 降級為 seuclidean:
    # seuclid = sqrt( (-0.2/0.5)^2 + (0.2/0.5)^2 ) = sqrt( 0.16 + 0.16 ) = sqrt(0.32) = 0.565685
    assert round(gsi_fallback, 4) == round(np.sqrt(0.32), 4)

def test_evaluate_isd_circuit_breaker():
    """
    1. 驗證 PASS 時，建議資金比與預估 Alpha 報酬率符合預期。
    2. 驗證當 status != 'PASS' (blocked) 時，suggestedPosition 被強制熔斷歸零 (0.0)。
    3. 驗證特徵包含 NaN 時，正確觸發 BLOCKED_BY_DQI_X。
    """
    latest_row_pass = pd.Series({
        'Z_Score_BIAS': -1.8,
        'K': 15.0, 'D': 16.0, 'KD_Passivation_Days': 4,
        'Feature_Range': 10.0, 'Feature_Slope': 1.2, 'Feature_STD': 2.5, 'Feature_SUM_ABS_X': 8.0,
        'Chip_Concentration_15': 0.1, 'Foreign_Net_Ratio': 0.2, 'Retail_Margin_Panic': 0.0, 'Big_Player_Force': 0.3
    })
    
    # CASE 1: 完美風控通過 (p_ref=0.72, p_conj=0.78, GSI=1.25, BigPlayerForce=0.3)
    # RI = 0.94 >= 0.60, GSI = 1.25 <= 3.00
    report_pass = evaluate_isd_circuit_breaker(0.72, 0.78, 1.25, latest_row_pass, 0.3)
    assert report_pass['status'] == "PASS"
    assert report_pass['isdTriggered'] is False
    assert report_pass['suggestedPosition'] == 0.80 # Tier 1: 80%
    # (0.78 - 0.5) * 0.186 = 0.05208
    assert report_pass['prediction']['predictedAlphaReturn'] == 0.0521
    
    # CASE 2: BLOCKED_BY_RI (RI = 0.40 < 0.60)
    report_ri = evaluate_isd_circuit_breaker(0.20, 0.80, 1.25, latest_row_pass, 0.3)
    assert report_ri['status'] == "BLOCKED_BY_RI"
    assert report_ri['isdTriggered'] is True
    assert report_ri['suggestedPosition'] == 0.0 # 熔斷強制歸零！
    
    # CASE 3: BLOCKED_BY_GSI (GSI = 3.50 > 3.00)
    report_gsi = evaluate_isd_circuit_breaker(0.72, 0.78, 3.50, latest_row_pass, 0.3)
    assert report_gsi['status'] == "BLOCKED_BY_GSI"
    assert report_gsi['isdTriggered'] is True
    assert report_gsi['suggestedPosition'] == 0.0 # 熔斷強制歸零！
    
    # CASE 4: BLOCKED_BY_DQI_X (K 線指標特徵損壞為 NaN)
    latest_row_nan = latest_row_pass.copy()
    latest_row_nan['K'] = np.nan
    report_dqix = evaluate_isd_circuit_breaker(0.72, 0.78, 1.25, latest_row_nan, 0.3)
    assert report_dqix['status'] == "BLOCKED_BY_DQI_X"
    assert report_dqix['isdTriggered'] is True
    assert report_dqix['suggestedPosition'] == 0.0 # 熔斷強制歸零！
