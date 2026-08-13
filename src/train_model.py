"""
=============================================================================
 src/train_model.py — AVM 閉環大腦 105 輪滾動 R2R 特訓營 (V3.5 上帝視角全勝版)
 升級重點：
   1. 日期格式強效歸一化：自動化解 YYYY-MM-DD 與 YYYY/MM/DD 匹配失效風險
   2. 冠軍基因全量灌載：讀取 `champion_trades.json` 6 大上帝視角勝率日 (Y=1, 3.0x 權重)
   3. 高檔逃頂錨點鎖定：同步灌入 6 大離場日 (Y=0, 2.0x 權重)，防止高檔過熱誤買
   4. 數值與標籤防禦：全量清洗 inf/NaN 與異常字串，保障 XGBoost 底層穩定運行
=============================================================================
"""

import sys
from pathlib import Path

# 動態解析專案根目錄並進行最高優先級的防重複安全注入，消除 ModuleNotFoundError 警告 (ISD 報告導出阻礙)
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

def execute_v34_closed_loop_rolling_train():
    print("🧠 [AVM 閉環大腦啟動] 正在加載黃金特徵矩陣...")
    
    current_dir = os.path.dirname(__file__)
    features_path = os.path.join(current_dir, "..", "data", "features_2379.csv")
    
    if not os.path.exists(features_path):
        print("❌ 找不到特徵矩陣檔案，請先執行 data_preprocessing.py！")
        return
        
    full_df = pd.read_csv(features_path)
    df = full_df[full_df['Target_Label'] != -1].copy()
    
    # 日期字串歸一化 (將 / 統一替換為 -，利於精準匹配)
    df['Date_Norm'] = df['Date'].astype(str).str.replace('/', '-').str.strip()
    
    X = df.drop(columns=['Date', 'Date_Norm', 'Target_Label'], errors='ignore')
    
    # ==========================================
    # 🛡️ 終極防彈裝甲：強制清洗所有異常數值
    # ==========================================
    X = X.apply(pd.to_numeric, errors='coerce')      # 強制轉為數字，怪異字串變 NaN
    X = X.replace([np.inf, -np.inf], np.nan)         # 將無限大 inf 轉為 NaN
    X = X.fillna(0)                                  # 將所有空值填為 0
    
    # 確保 y 標籤絕對乾淨 (0 與 1)
    if 'Target_Label' in df.columns:
        y = pd.to_numeric(df['Target_Label'], errors='coerce').fillna(0).astype(int)
        y = np.where(y > 0, 1, 0)
        y = pd.Series(y, index=df.index)
    else:
        print("❌ 找不到 Target_Label！")
        return
    # ==========================================
    
    total_days = len(df)
    warmup_days = 60   # 60 天冷啟動視窗 (V3.8)
    step_days = 5      # 1 週滾動修正步長
    
    loop_range = list(range(warmup_days, total_days, step_days))
    total_loops = len(loop_range)
    
    feedback_multipliers = np.ones(total_days)
    
    # 👑【老闆實戰黃金教案定錨】
    golden_dates = ['2026-03-05', '2026-01-26', '2026-03-04']
    for g_date in golden_dates:
        golden_mask = df['Date_Norm'].str.contains(g_date)
        if golden_mask.any():
            golden_indices = df[golden_mask].index.tolist()
            for g_idx in golden_indices:
                y.loc[g_idx] = 1            # 強制設定為正向買點 (Y = 1)
                feedback_multipliers[g_idx] *= 3.0
            print(f"👑 [老闆黃金教案鎖定] 已將實戰戰果日 {g_date} 灌入大腦 (Y=1)，訓練權重提升至 3.0倍！")

    # 🏆【讀取上帝視角冠軍基因檔】
    champion_file = os.path.join(current_dir, "..", "data", "champion_trades.json")
    if os.path.exists(champion_file):
        try:
            with open(champion_file, 'r', encoding='utf-8') as f:
                champ_data = json.load(f)
                winning_dates = champ_data.get('winning_dates', [])
                exit_dates = champ_data.get('exit_dates', [])
                
                # 1. 加強 6 大勝率波段進場日 (Y = 1, 權重 3.0x)
                w_count = 0
                for c_date in winning_dates:
                    c_mask = df['Date_Norm'].str.contains(c_date)
                    if c_mask.any():
                        c_indices = df[c_mask].index.tolist()
                        for c_idx in c_indices:
                            y.loc[c_idx] = 1
                            feedback_multipliers[c_idx] *= 3.0
                            w_count += 1
                
                # 2. 鎖定 6 大逃頂高點離場日 (Y = 0, 權重 2.0x)
                e_count = 0
                for e_date in exit_dates:
                    e_mask = df['Date_Norm'].str.contains(e_date)
                    if e_mask.any():
                        e_indices = df[e_mask].index.tolist()
                        for e_idx in e_indices:
                            y.loc[e_idx] = 0
                            feedback_multipliers[e_idx] *= 2.0
                            e_count += 1
                            
                print(f"🏆 [冠軍基因回饋成功] 讀取【{champ_data.get('name')}】！")
                print(f"   └─ 已強化 {w_count} 個進場波段買點 (Y=1, 3.0x 權重)")
                print(f"   └─ 已鎖定 {e_count} 個高檔逃頂賣點 (Y=0, 2.0x 權重)")
        except Exception as e:
            print(f"⚠️ 讀取冠軍基因檔失敗: {e}")
            
    # 建立 models 備份目錄 (V3.8)
    models_dir = os.path.join(current_dir, "..", "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    gsi_baseline_path = os.path.join(current_dir, "..", "data", "gsi_baseline.npz")
    output_conj_path = os.path.join(current_dir, "..", "data", "model_2379_conj.json")
    output_main_path = os.path.join(current_dir, "..", "data", "model_2379.json")
    output_ref_path = os.path.join(current_dir, "..", "data", "model_2379_ref.json")
    
    all_y_test = []
    all_y_pred = []
    
    # 進入高頻週滾動反饋控制與 GSI 演進迴圈 (V3.8)
    for idx, current_step in enumerate(loop_range, 1):
        end_test = min(current_step + step_days, total_days)
        
        X_train, X_test = X.iloc[:current_step], X.iloc[current_step:end_test]
        y_train, y_test = y.iloc[:current_step], y.iloc[current_step:end_test]
        
        if len(X_test) == 0:
            break
            
        distances = (current_step - 1) - np.arange(current_step)
        time_decay_weights = np.exp(-0.0025 * distances)
        current_sample_weights = time_decay_weights * feedback_multipliers[:current_step]
        
        # 重新 Fit XGBoost 模型 M_conj
        model_conj = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric='logloss',
            random_state=42
        )
        model_conj.fit(X_train, y_train, sample_weight=current_sample_weights)
        
        # 雙軌快取: 歸檔冷路徑至 data/models/model_2379_w{week_idx:02d}_conj.json
        archive_path = os.path.join(models_dir, f"model_2379_w{idx:02d}_conj.json")
        model_conj.save_model(archive_path)
        
        # 雙軌快取: 覆寫熱路徑
        model_conj.save_model(output_conj_path)
        model_conj.save_model(output_main_path)
        model_conj.save_model(output_ref_path) # 雙模型對稱定錨
        
        # Recalculate GSI Baseline 均值 mu 與逆協方差矩陣 Σ^-1 (包含 Ridge 正則化與 Pseudo-inverse)
        mean_vec = np.mean(X_train, axis=0).values
        cov_mat = np.cov(X_train, rowvar=False)
        std_vec = np.std(X_train, axis=0).values
        
        I = np.eye(cov_mat.shape[0])
        cov_reg = cov_mat + 1e-6 * I
        inv_cov = np.linalg.pinv(cov_reg)
        
        np.savez(gsi_baseline_path, mean=mean_vec, inv_cov=inv_cov, std=std_vec)
        
        # 預測
        preds = model_conj.predict(X_test)
        
        # R2R 閉環修正：挨打懊悔機制
        for i, (actual, pred) in enumerate(zip(y_test, preds)):
            global_idx = current_step + i
            if actual != pred:
                feedback_multipliers[global_idx] = 2.5
        
        all_y_test.extend(y_test)
        all_y_pred.extend(preds)
        
        if idx % 10 == 0 or idx == total_loops:
            print(f"⏳ [大腦特訓中] 已通關第 {idx:3d}/{total_loops} 輪地獄滾動補考... (進度: {idx/total_loops*100:.1f}%)")
    
    print(f"\n🏁 {total_loops} 輪動態閉環反思特訓全線通關！大腦已完成全歷史的挨打與自我更正。")
    print("\n" + "═"*50)
    print(f"📊 【完全體 AVM 自我迭代大腦 ➔ {total_loops} 輪歷史累積盲測報告】")
    print("═"*50)
    
    cm = confusion_matrix(all_y_test, all_y_pred)
    total_signals = cm[0, 1] + cm[1, 1]
    final_precision = (cm[1, 1] / total_signals) * 100 if total_signals > 0 else 0
    
    print(f"🎯 歷史跨越盲測期 ➔ 實戰摸底綜合勝率：{final_precision:.1f} %")
    print(f"📈 實體作戰戰果 ➔ 成功抓到強勢V轉：{cm[1, 1]} 次")
    print(f"📉 慘遭主力洗盤 ➔ 被假摔甩轎騙進去：{cm[0, 1]} 次")
    print(f"🔍 漏網之魚統計 ➔ 歷史上有 {cm[1, 0]} 次 V 轉行情 AI 選擇保守觀望未出手")
    print("═"*50)
    
    # =========================================================================
    # 👑 全量數據最終熔煉與雙模型對稱定錨 (V3.8: 整合最新一期預測報告產出)
    # =========================================================================
    final_total_days = len(df)
    
    # 1. 訓練 M_ref (長期參考模型)：3年歷史全量，低衰減 (0.0005)
    print(f"🛡️ 正在訓練長期參考模型 M_ref (衰減率: 0.0005)...")
    ref_distances = (final_total_days - 1) - np.arange(final_total_days)
    ref_time_decay = np.exp(-0.0005 * ref_distances)
    ref_weights = ref_time_decay * feedback_multipliers[:final_total_days]
    
    model_ref = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model_ref.fit(X, y, sample_weight=ref_weights)
    
    # 2. 訓練 M_conj (短期推測模型)：近 120 天資料，高衰減 (0.0025)
    conj_days = min(120, final_total_days)
    print(f"🚀 正在訓練短期推測模型 M_conj (取近 {conj_days} 天，衰減率: 0.0025)...")
    X_conj = X.iloc[-conj_days:]
    y_conj = y.iloc[-conj_days:]
    
    conj_distances = (conj_days - 1) - np.arange(conj_days)
    conj_time_decay = np.exp(-0.0025 * conj_distances)
    conj_weights = conj_time_decay * feedback_multipliers[-conj_days:]
    
    model_conj = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model_conj.fit(X_conj, y_conj, sample_weight=conj_weights)
    
    # 輸出 M_conj (最新大腦) 決策特徵重要性排行榜
    importance = model_conj.feature_importances_
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    print("\n🔍 【短期推測大腦 M_conj 決策權重 (Feature Importance 排行榜)】：")
    for feat, imp in feat_imp.head(8).items():
        print(f"  - {feat:<22}: {imp*100:.2f}%")
        
    model_ref.save_model(output_ref_path)
    model_conj.save_model(output_conj_path)
    model_conj.save_model(output_main_path)
    print(f"📂 雙模型存檔成功：")
    print(f"   ├─ M_ref  -> {output_ref_path}")
    print(f"   └─ M_conj -> {output_conj_path} (已同步複製至 model_2379.json)\n")
    
    # =========================================================================
    # 📈 計算並匯出 GSI Baseline 指標基底 (馬氏距離背景參照數據)
    # =========================================================================
    print("📈 正在計算並匯出 GSI Baseline 指標基底 (120天特徵常態空間)...")
    X_120 = X.iloc[-120:] if len(X) >= 120 else X
    mean_vec = np.mean(X_120, axis=0).values
    cov_mat = np.cov(X_120, rowvar=False)
    std_vec = np.std(X_120, axis=0).values
    
    I = np.eye(cov_mat.shape[0])
    cov_reg = cov_mat + 1e-6 * I
    inv_cov = np.linalg.pinv(cov_reg)
    
    np.savez(gsi_baseline_path, mean=mean_vec, inv_cov=inv_cov, std=std_vec)
    print(f"📂 GSI Baseline 數據已成功匯出至：{gsi_baseline_path}\n")

    # 3. 呼叫 predict_realtime_v_turn 導出最新 DQIx/ISD 熔斷診斷報告 (V3.8)
    try:
        from src.predict_today import predict_realtime_v_turn
        predict_realtime_v_turn()
    except Exception as e:
        print(f"⚠️ [ISD 報告導出阻礙] {e}")

if __name__ == "__main__":
    execute_v34_closed_loop_rolling_train()