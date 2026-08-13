import os
import json
import datetime
from typing import List
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# 自適應載入相同目錄下的 GSI 計算
from src.predict_today import calculate_gsi

def evaluate_pivot_signal(action_type: str, p_conj: float, z_bias: float) -> str:
    """
    轉折點訊號標註判定邏輯：
    - BUY 點：若 p_conj >= 0.45，標註為 [SIGNAL_OK]；否則為 [MODEL_BLIND]。
    - SELL 點：若 z_bias >= 2.0 或 p_conj < 0.40，標註為 [SIGNAL_OK]；否則為 [MODEL_BLIND]。
    """
    if action_type == "BUY":
        if p_conj >= 0.45:
            return "[SIGNAL_OK]"
        else:
            return "[MODEL_BLIND]"
    elif action_type == "SELL":
        if z_bias >= 2.0 or p_conj < 0.40:
            return "[SIGNAL_OK]"
        else:
            return "[MODEL_BLIND]"
    return "[MODEL_BLIND]"

def get_nearest_trading_date(target_date_str: str, available_dates: List[str]) -> str:
    """
    若目標日期遇到非交易日，自動往前（過去時間）尋找最近的一個交易日。
    """
    target_dt = pd.to_datetime(target_date_str)
    # 尋找所有小於等於目標日期的交易日
    valid_dates = [d for d in available_dates if pd.to_datetime(d) <= target_dt]
    if not valid_dates:
        return available_dates[0]
    return valid_dates[-1]

def run_signal_audit():
    print("📡 [轉折點稽核啟動] 正在加載雙大腦模型與歷史特徵...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 1. 定位並讀取資料與模型路徑
    feat_path = os.path.join(project_root, "data", "features_2379.csv")
    model_ref_path = os.path.join(project_root, "data", "model_2379_ref.json")
    model_conj_path = os.path.join(project_root, "data", "model_2379_conj.json")
    gsi_baseline_path = os.path.join(project_root, "data", "gsi_baseline.npz")
    champion_path = os.path.join(project_root, "data", "champion_trades.json")
    
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"❌ 找不到特徵矩陣 {feat_path}！")
    if not os.path.exists(model_ref_path) or not os.path.exists(model_conj_path):
        raise FileNotFoundError("❌ 缺少雙週期大腦模型 json 檔案！請先執行 train_model.py")
    if not os.path.exists(gsi_baseline_path):
        raise FileNotFoundError("❌ 缺少 GSI Baseline 基底數據！")
    if not os.path.exists(champion_path):
        raise FileNotFoundError("❌ 缺少 上帝視角冠軍基因檔案 champion_trades.json！")
        
    # 2. 載入雙模型與 GSI Baseline
    model_ref = XGBClassifier()
    model_ref.load_model(model_ref_path)
    
    model_conj = XGBClassifier()
    model_conj.load_model(model_conj_path)
    
    baseline = np.load(gsi_baseline_path)
    mean_vec = baseline['mean']
    inv_cov = baseline['inv_cov']
    std_vec = baseline['std'] if 'std' in baseline.files else None
    
    # 3. 載入 12 個轉折日日期
    with open(champion_path, "r", encoding="utf-8") as f:
        champ_data = json.load(f)
    
    winning_dates = champ_data.get("winning_dates", []) # 6 BUY 點
    exit_dates = champ_data.get("exit_dates", [])       # 6 SELL 點
    
    # 4. 讀取特徵檔案並歸一化日期
    df_feat = pd.read_csv(feat_path)
    df_feat['Date_Str'] = pd.to_datetime(df_feat['Date']).dt.strftime('%Y-%m-%d')
    available_dates = sorted(df_feat['Date_Str'].unique().tolist())
    
    feature_cols = [
        'Z_Score_BIAS', 'K', 'D', 'KD_Passivation_Days', 
        'Feature_Range', 'Feature_Slope', 'Feature_STD', 'Feature_SUM_ABS_X',
        'Chip_Concentration_15', 'Foreign_Net_Ratio', 'Retail_Margin_Panic', 'Big_Player_Force',
        'foreign_net_buy', 'invest_net_buy', 'dealer_net_buy', 'margin_balance_chg', 'major_chip_ratio',
        'atr_14'
    ]
    
    audit_results = []
    
    # 5. 執行 12 轉折日特徵比對與標註
    all_targets = [(d, "BUY") for d in winning_dates] + [(d, "SELL") for d in exit_dates]
    # 依日期排序，保持歷史順序
    all_targets.sort(key=lambda x: x[0])
    
    for target_date, action_type in all_targets:
        # 自適應非交易日往前對位
        actual_date = get_nearest_trading_date(target_date, available_dates)
        
        row_feat = df_feat[df_feat['Date_Str'] == actual_date].iloc[0]
        x_array = row_feat[feature_cols].apply(pd.to_numeric).values
        df_row_df = pd.DataFrame([x_array], columns=feature_cols)
        
        # 雙大腦勝率預估
        p_ref = float(model_ref.predict_proba(df_row_df)[0][1])
        p_conj = float(model_conj.predict_proba(df_row_df)[0][1])
        
        # 計算 GSI
        gsi_val = calculate_gsi(x_array, mean_vec, inv_cov, std_vec)
        
        # Z_BIAS
        z_bias = float(row_feat['Z_Score_BIAS'])
        
        # 標註判定
        status = evaluate_pivot_signal(action_type, p_conj, z_bias)
        
        audit_results.append({
            "target_date": target_date,
            "actual_trading_date": actual_date,
            "action_type": action_type,
            "p_conj": round(p_conj, 4),
            "p_ref": round(p_ref, 4),
            "z_bias": round(z_bias, 4),
            "gsi": round(gsi_val, 4),
            "status": status
        })
        
    # 6. 打包寫入 audit_signals_report.json 
    report_output = {
        "timestamp": datetime.datetime.now().isoformat()[:19],
        "audit_results": audit_results
    }
    
    report_output_path = os.path.join(project_root, "data", "audit_signals_report.json")
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2, ensure_ascii=False)
    print(f"📂 轉折點審計報告已成功導出至：{report_output_path}\n")
    
    # 7. 格式化 Console 輸出
    print(f"\n{'=' * 105}")
    print(f"  📊 AVM 雙大腦 12 大黃金轉折日訊號審計報告 (V3.6)")
    print(f"{'=' * 105}")
    print(f"  {'對位日':<12} {'實際交易日':<12} {'動作':<6} {'pConj':<8} {'pRef':<8} {'Z_BIAS':<8} {'GSI':<8} {'稽核結果':<15}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 15}")
    
    for r in audit_results:
        print(f"  {r['target_date']:<12} {r['actual_trading_date']:<12} {r['action_type']:<6} {r['p_conj']:<8.4f} {r['p_ref']:<8.4f} {r['z_bias']:<8.4f} {r['gsi']:<8.4f} {r['status']:<15}")
    print(f"{'=' * 105}\n")

if __name__ == "__main__":
    run_signal_audit()
