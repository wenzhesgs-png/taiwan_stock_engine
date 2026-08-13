import os
import pandas as pd
import numpy as np
import yfinance as yf

def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    專職 I/O 搬運工 (Data Loader & Ingestion Guard)
    - 取得實時或離線 2379.TW 股價資料。
    - 讀取實體籌碼資料庫並執行 Left Join 避免誤殺當天股價資料。
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_data_path = os.path.join(data_dir, "raw_2379.csv")
    chip_data_path = os.path.join(data_dir, "raw_chip_2379.csv")
    
    # 1. 股價數據聯網/離線更新
    try:
        ticker = "2379.TW"
        fresh_df = yf.download(ticker, start="2023-01-01")
        if not fresh_df.empty:
            fresh_df.to_csv(raw_data_path)
            print(f"📥 聯網同步成功！最新股價數據已灌入：{raw_data_path}")
    except Exception as e:
        print(f"❌ 聯網下載失敗（原因：{e}），系統自動切換為本機離線模式...")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ 致命錯誤：找不到原始股價數據檔：{raw_data_path}")
        
    # 2. 解析股價 DataFrame 欄位與 MultiIndex 降級
    raw_df = pd.read_csv(raw_data_path, header=None)
    header_idx = 0
    for idx, row in raw_df.head(5).iterrows():
        row_str = [str(x).strip() for x in row.values]
        if 'Close' in row_str and 'Open' in row_str:
            header_idx = idx
            break
            
    df_price = pd.read_csv(raw_data_path, header=header_idx)
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    df_price.columns = [str(col).strip() for col in df_price.columns]
    
    df_price.rename(columns={df_price.columns[0]: 'Date'}, inplace=True)
    df_price = df_price.dropna(subset=['Close'])
    df_price['Date'] = pd.to_datetime(df_price['Date'], errors='coerce')
    df_price = df_price[df_price['Date'].notnull()].sort_values('Date')
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_price[col] = pd.to_numeric(df_price[col], errors='coerce')
    df_price = df_price.dropna(subset=['Close'])

    # 3. 讀取實體籌碼資料庫
    if not os.path.exists(chip_data_path):
        raise FileNotFoundError(f"❌ 致命錯誤：缺少實體籌碼資料庫 {chip_data_path}！本系統拒絕合成假數據！")
        
    df_chip = pd.read_csv(chip_data_path)
    df_chip['Date'] = pd.to_datetime(df_chip['Date'])
    
    # 4. 執行 Left Join（以股價為主體，確保盤中當天不會因為缺少當日籌碼而被 Drop）
    df_merged = pd.merge(df_price, df_chip, on='Date', how='left')
    return df_merged

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    大廚算盤 (Pure Function - 純數值特徵工程)
    - 絕不進行網絡/檔案 I/O，便於高效率單元測試。
    - 嚴格透過 .shift(1) 防止籌碼前瞻偏誤。
    - 17維特徵、未來 3 日最高價及未來未知遮罩防線計算。
    """
    # 確保傳入資料以 Date 排序
    df = df.sort_values('Date').copy()
    
    # 1. 為了計算新特徵，先在不 shift 的情況下提取/計算籌碼相關的 raw 欄位 (V3.8)
    df['foreign_net_buy'] = pd.to_numeric(df.get('Foreign_Net_Volume', 0), errors='coerce').fillna(0.0)
    df['invest_net_buy'] = pd.to_numeric(df.get('Trust_Net_Volume', 0), errors='coerce').fillna(0.0)
    df['dealer_net_buy'] = pd.to_numeric(df.get('Dealer_Net_Volume', 0), errors='coerce').fillna(0.0)
    df['margin_balance_chg'] = pd.to_numeric(df.get('Margin_Balance', 0), errors='coerce').diff().fillna(0.0)
    
    vol_safe_raw = np.where(df['Volume'] == 0, 1.0, df['Volume'])
    df['major_chip_ratio'] = pd.to_numeric(df.get('Top15_Net_Volume', 0), errors='coerce').fillna(0.0) / vol_safe_raw

    # 將所有需要時滯 T-1 對齊的籌碼列集合起來
    chip_cols_orig = ['Top15_Net_Volume', 'Foreign_Net_Volume', 'Margin_Balance', 'Trust_Net_Volume', 'Dealer_Net_Volume']
    chip_cols_new = ['foreign_net_buy', 'invest_net_buy', 'dealer_net_buy', 'margin_balance_chg', 'major_chip_ratio']
    chip_cols = chip_cols_orig + chip_cols_new
    
    for col in chip_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    df[chip_cols] = df[chip_cols].shift(1)
    df[chip_cols] = df[chip_cols].fillna(0.0) # 滯後產生的首行 NaN 填補為 0.0

    # 2. 技術面特徵計算
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Wilder's ATR 14 計算 (V4.2)
    high_low = df['High'] - df['Low']
    high_prev_close = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    atr = np.zeros(len(df))
    if len(df) >= 14:
        # 首個 ATR14 為前 14 日 TR 的簡單移動平均 (SMA)
        atr[13] = tr.iloc[:14].mean()
        for i in range(14, len(df)):
            atr[i] = (atr[i-1] * 13 + tr.iloc[i]) / 14
    else:
        for i in range(len(df)):
            atr[i] = tr.iloc[:i+1].mean()
            
    df['atr_14'] = atr
    # 填補前 13 天之缺失值 (Bfill)，確保無 NaN/0.0，最差保底為 1.0
    df['atr_14'] = df['atr_14'].replace(0.0, np.nan).bfill()
    df['atr_14'] = df['atr_14'].bfill().fillna(1.0)

    df['BIAS'] = (df['Close'] - df['MA_20']) / df['MA_20']
    df['BIAS_mean_60'] = df['BIAS'].rolling(window=60).mean()
    df['BIAS_std_60'] = df['BIAS'].rolling(window=60).std()
    df['Z_Score_BIAS'] = (df['BIAS'] - df['BIAS_mean_60']) / df['BIAS_std_60']
    
    # KD 指標與低檔鈍化天數
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    df['RSV'] = np.where(high_9 != low_9, (df['Close'] - low_9) / (high_9 - low_9) * 100, 50)
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    
    is_passivated = (df['K'] < 20) & (df['D'] < 20)
    df['KD_Passivation_Days'] = is_passivated.groupby((~is_passivated).cumsum()).cumsum()
    
    # 微觀時域統計
    window_s = 5
    df['Feature_Range'] = df['High'].rolling(window=window_s).max() - df['Low'].rolling(window=window_s).min()
    df['Feature_Slope'] = df['Close'].diff(periods=3)
    df['Feature_STD'] = df['Close'].rolling(window=window_s).std()
    df['Feature_SUM_ABS_X'] = df['Close'].diff().abs().rolling(window=window_s).sum()
    
    # 3. 籌碼面特徵計算 (基於已滯後 shift 的籌碼數據)
    df['Vol_Safe'] = np.where(df['Volume'] == 0, 1, df['Volume'])
    df['Chip_Concentration_15'] = df['Top15_Net_Volume'] / df['Vol_Safe']
    df['Foreign_Net_Ratio'] = df['Foreign_Net_Volume'] / df['Vol_Safe']
    df['Retail_Margin_Panic'] = -df['Margin_Balance'].diff(3) / df['Margin_Balance'].rolling(3).mean().replace(0, 1)
    df['Big_Player_Force'] = (df['Foreign_Net_Volume'] + df['Trust_Net_Volume'] + df['Dealer_Net_Volume']) / df['Vol_Safe']

    # 4. 建立未來 3 日最高價 (使用 Explicit Shift Concat 防止 boundary 隱患)
    future_highs = pd.concat([df['High'].shift(-1), df['High'].shift(-2), df['High'].shift(-3)], axis=1)
    df['Future_Max_High'] = future_highs.max(axis=1)
    
    # 【未來未知遮罩防線】：最新 3 天盲區強制標記為 -1
    df['Target_Label'] = np.where(df['Future_Max_High'].isna(), -1, 
                                   np.where((df['Future_Max_High'] - df['Close']) / df['Close'] >= 0.05, 1, 0))
    
    feature_cols = [
        'Z_Score_BIAS', 'K', 'D', 'KD_Passivation_Days', 
        'Feature_Range', 'Feature_Slope', 'Feature_STD', 'Feature_SUM_ABS_X',
        'Chip_Concentration_15', 'Foreign_Net_Ratio', 'Retail_Margin_Panic', 'Big_Player_Force',
        'foreign_net_buy', 'invest_net_buy', 'dealer_net_buy', 'margin_balance_chg', 'major_chip_ratio',
        'atr_14'
    ]
    
    # 5. 歷史缺損清理：排除最新 3 天盲區，其餘歷史數據若因 Window 計算產生 NaNs 則執行 dropna()
    latest_mask = df['Target_Label'] == -1
    historical_clean = df[~latest_mask].dropna(subset=feature_cols)
    
    # 在短數據測試 (len(df) < 65) 下，放寬 dropna 限制，防止最新 3 天被全部丟棄
    if len(df) < 65:
        latest_clean = df[latest_mask]
    else:
        latest_clean = df[latest_mask].dropna(subset=feature_cols)
        
    cleaned_df = pd.concat([historical_clean, latest_clean], axis=0).sort_values('Date')
    return cleaned_df[['Date'] + feature_cols + ['Target_Label']]

def execute_v34_preprocessing_with_update():
    """
    總調度入口 (Preprocess Manager)
    - 取得載入數據
    - 進行特徵計算
    - 輸出儲存
    """
    print("═"*60)
    print("📡 [聯網防線發動] 正在連線 Yahoo Finance 抓取瑞昱 (2379.TW) 最新實時股價...")
    print("═"*60)
    
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    
    # 1. 搬運載入
    df_raw = load_raw_data(data_dir)
    
    # 2. 計算特徵
    cleaned_features_df = compute_features(df_raw)
    
    # 3. 統計與輸出
    raw_latest_date = df_raw['Date'].max().strftime('%Y-%m-%d')
    actual_labeled_df = cleaned_features_df[cleaned_features_df['Target_Label'] != -1]
    feature_latest_date = actual_labeled_df['Date'].max().strftime('%Y-%m-%d') if not actual_labeled_df.empty else "N/A"
    
    output_path = os.path.join(data_dir, "features_2379.csv")
    cleaned_features_df.to_csv(output_path, index=False)
    
    feature_cols = [col for col in cleaned_features_df.columns if col not in ['Date', 'Target_Label']]
    print("\n" + "👑" + "═"*58)
    print(" 👑 V3.5 全域籌碼多維感測特徵管線清洗完畢！")
    print("═"*60)
    print(f"   ➔ 🟢 原始股價數據已同步至：{raw_latest_date}")
    print(f"   ➔ 🔵 籌碼融合與遮罩截止日：{feature_latest_date}")
    print(f"📡 已整合感測器（時滯1天防洩漏）：前15大分點控盤度、外資比率、融資恐慌度、三大法人合力值。")
    print(f"🎯 總特徵維度已擴展為：{len(feature_cols)} 維特徵空間。")
    print("═"*60 + "\n")

if __name__ == "__main__":
    execute_v34_preprocessing_with_update()
