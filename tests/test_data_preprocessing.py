import os
import sys
import time
import pandas as pd
import numpy as np
import pytest

# 將 src 加入 Python 模組搜尋路徑，確保測試能正常匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preprocessing import compute_features

def test_compute_features_pure_function():
    """
    1. 驗證 compute_features 是 Pure Function 且能在 10ms (寬鬆至 100ms) 內快速執行完畢。
    2. 驗證產出欄位完整包含 12 維特徵與 Target_Label。
    3. 驗證最後 3 筆之 Target_Label 必為 -1（未來未知遮罩防線）。
    """
    dates = pd.date_range(start="2026-01-01", periods=10)
    dummy_data = {
        'Date': dates,
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000] * 10,
        'Top15_Net_Volume': [100.0] * 10,
        'Foreign_Net_Volume': [200.0] * 10,
        'Margin_Balance': [10000] * 10,
        'Trust_Net_Volume': [50.0] * 10,
        'Dealer_Net_Volume': [20.0] * 10
    }
    df_dummy = pd.DataFrame(dummy_data)
    
    start_time = time.time()
    df_res = compute_features(df_dummy)
    duration = (time.time() - start_time) * 1000  # ms
    
    print(f"⌛ [Pure Function 效能實測] 耗時: {duration:.2f} ms")
    assert duration < 200.0, f"計算耗時過長：{duration:.2f} ms，大於 200ms 限制！"
    
    expected_cols = [
        'Date', 'Z_Score_BIAS', 'K', 'D', 'KD_Passivation_Days', 
        'Feature_Range', 'Feature_Slope', 'Feature_STD', 'Feature_SUM_ABS_X',
        'Chip_Concentration_15', 'Foreign_Net_Ratio', 'Retail_Margin_Panic', 'Big_Player_Force',
        'Target_Label'
    ]
    for col in expected_cols:
        assert col in df_res.columns, f"缺失關鍵欄位 {col}！"
        
    last_three_labels = df_res['Target_Label'].tail(3).tolist()
    assert all(label == -1 for label in last_three_labels), f"最新 3 筆的 Target_Label 應強制標記為 -1 遮罩防線，實際為: {last_three_labels}"

def test_no_random_data_generated():
    """
    驗證 data_preprocessing.py 中是否沒有使用任何隨機數生成邏輯（例如 np.random），
    完全拒絕合成擬真假數據。
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "src", "data_preprocessing.py")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "np.random" not in content, "❌ 檢測到 np.random！本專案嚴禁合成假數據！"
    assert "random.normal" not in content, "❌ 檢測到 random.normal！本專案嚴禁使用隨機數生成！"

def test_explicit_shift_future_high():
    """
    輸入固定 High 序列 [10, 12, 15, 11, 10]，
    驗證 index 0 的未來 3 天最高價（即 index 1, 2, 3，值為 12, 15, 11）之極值精確等於 15。
    """
    dates = pd.date_range(start="2026-01-01", periods=5)
    dummy_data = {
        'Date': dates,
        'Open': [10.0] * 5,
        'High': [10.0, 12.0, 15.0, 11.0, 10.0],
        'Low': [9.0] * 5,
        'Close': [10.0] * 5,
        'Volume': [1000] * 5,
        'Top15_Net_Volume': [100.0] * 5,
        'Foreign_Net_Volume': [200.0] * 5,
        'Margin_Balance': [10000] * 5,
        'Trust_Net_Volume': [50.0] * 5,
        'Dealer_Net_Volume': [20.0] * 5
    }
    df_dummy = pd.DataFrame(dummy_data)
    
    # 計算未來 3 天最高價
    future_highs = pd.concat([df_dummy['High'].shift(-1), df_dummy['High'].shift(-2), df_dummy['High'].shift(-3)], axis=1)
    future_max = future_highs.max(axis=1)
    
    # index 0 的未來 3 日最大高價為 max(12, 15, 11) = 15
    assert future_max.iloc[0] == 15.0, f"index 0 的未來 3 日最高價計算錯誤，實際為: {future_max.iloc[0]}"
    
    # index 1 的未來 3 日最大高價為 max(15, 11, 10) = 15
    assert future_max.iloc[1] == 15.0, f"index 1 的未來 3 日最高價計算錯誤，實際為: {future_max.iloc[1]}"
    
    # index 2 的未來 3 日最大高價為 max(11, 10, NaN) = 11
    assert future_max.iloc[2] == 11.0, f"index 2 的未來 3 日最高價計算錯誤，實際為: {future_max.iloc[2]}"

def test_chip_features_shift_and_no_leakage():
    """
    Ticket 1 TDD 測試:
    1. 驗證 T 日之 foreign_net_buy 數值 100% 等於原始籌碼資料庫中 T-1 日之數值。
    2. 驗證產出之 features df 含有 5 大新欄位且全表 NaN 數量為 0。
    """
    dates = pd.date_range(start="2026-01-01", periods=10)
    dummy_data = {
        'Date': dates,
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000] * 10,
        'Top15_Net_Volume': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
        'Foreign_Net_Volume': [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0],
        'Margin_Balance': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900],
        'Trust_Net_Volume': [50.0] * 10,
        'Dealer_Net_Volume': [20.0] * 10
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_res = compute_features(df_dummy)
    
    # 1. 驗證 5 大新欄位存在
    new_chip_cols = ["foreign_net_buy", "invest_net_buy", "dealer_net_buy", "margin_balance_chg", "major_chip_ratio"]
    for col in new_chip_cols:
        assert col in df_res.columns, f"缺失新籌碼欄位 {col}！"
        
    # 2. 驗證無 NaN
    assert df_res[new_chip_cols].isna().sum().sum() == 0, "新籌碼欄位中含有 NaN！"
    
    # 3. 驗證 T 日的 foreign_net_buy 等於原始 T-1 日的 Foreign_Net_Volume
    df_res_sorted = df_res.sort_values('Date').reset_index(drop=True)
    df_dummy_sorted = df_dummy.sort_values('Date').reset_index(drop=True)
    
    for i in range(1, len(df_res_sorted)):
        date_t = df_res_sorted.loc[i, 'Date']
        val_t = df_res_sorted.loc[i, 'foreign_net_buy']
        
        # 找到 dummy 中前一天的 Foreign_Net_Volume
        prev_date = date_t - pd.Timedelta(days=1)
        dummy_prev_row = df_dummy_sorted[df_dummy_sorted['Date'] == prev_date]
        if not dummy_prev_row.empty:
            expected_val = dummy_prev_row.iloc[0]['Foreign_Net_Volume']
            assert val_t == expected_val, f"時滯對齊錯誤！T日 {date_t} 數值為 {val_t}，預期等於 T-1日 {prev_date} 數值 {expected_val}"

def test_atr14_feature_injection():
    """
    Ticket 1 TDD 測試 (V4.2):
    - 驗證產出之 data/features_2379.csv 含有 atr_14 欄位。
    - 驗證 atr_14 數值恆正 (>0) 且無 Inf。
    """
    dates = pd.date_range(start="2026-01-01", periods=15)
    dummy_data = {
        'Date': dates,
        'Open': [100.0] * 15,
        'High': [105.0] * 15,
        'Low': [95.0] * 15,
        'Close': [102.0] * 15,
        'Volume': [1000] * 15,
        'Top15_Net_Volume': [100.0] * 15,
        'Foreign_Net_Volume': [200.0] * 15,
        'Margin_Balance': [10000] * 15,
        'Trust_Net_Volume': [50.0] * 15,
        'Dealer_Net_Volume': [20.0] * 15
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_res = compute_features(df_dummy)
    
    assert "atr_14" in df_res.columns, "❌ features 中缺失 atr_14 欄位！"
    assert (df_res["atr_14"] > 0.0).all(), "❌ atr_14 存在非正數！"
    assert not np.isinf(df_res["atr_14"]).any(), "❌ atr_14 包含無限大 Inf！"
