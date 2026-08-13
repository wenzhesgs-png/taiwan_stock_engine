import os
import json
import datetime
import pandas as pd
import numpy as np
import sys

# 將 src 加入 Python 搜尋路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulate_agents import load_data_and_predict_brain

def run_brain_accuracy_audit():
    print("📡 [大腦預測力審計啟動] 正在加載特徵與預測勝率數據...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 1. 載入特徵與大腦預測勝率 (AI_Probability)
    df = load_data_and_predict_brain()
    
    # 2. 計算未來 N 日真實收益率 (N in {3, 5, 10})
    # R_{t+N} = (Close_{t+N} - Close_t) / Close_t
    for N in [3, 5, 10]:
        df[f'Return_{N}d'] = (df['Close'].shift(-N) - df['Close']) / df['Close']
        
    # 3. 計算 Pearson Correlation (IC_N)
    ic_3d = df['AI_Probability'].corr(df['Return_3d'], method='pearson')
    ic_5d = df['AI_Probability'].corr(df['Return_5d'], method='pearson')
    ic_10d = df['AI_Probability'].corr(df['Return_10d'], method='pearson')
    
    # 處理 NaN (如果數據量少或全為常數導致 corr 變為 NaN)
    ic_3d = 0.0 if np.isnan(ic_3d) or np.isinf(ic_3d) else float(ic_3d)
    ic_5d = 0.0 if np.isnan(ic_5d) or np.isinf(ic_5d) else float(ic_5d)
    ic_10d = 0.0 if np.isnan(ic_10d) or np.isinf(ic_10d) else float(ic_10d)
    
    # 4. 計算 Directional Precision (p_conj >= 0.70 時, 未來 5 日收益率 > 0 的成功率)
    high_confidence_df = df[df['AI_Probability'] >= 0.70]
    if not high_confidence_df.empty:
        directional_precision_70 = (high_confidence_df['Return_5d'] > 0).mean()
        directional_precision_70 = 0.0 if np.isnan(directional_precision_70) or np.isinf(directional_precision_70) else float(directional_precision_70)
    else:
        directional_precision_70 = 0.0
        
    # 5. 打包寫入 data/brain_accuracy_report.json
    report = {
        "timestamp": datetime.datetime.now().isoformat()[:19],
        "ic_3d": round(ic_3d, 4),
        "ic_5d": round(ic_5d, 4),
        "ic_10d": round(ic_10d, 4),
        "directional_precision_70": round(directional_precision_70, 4)
    }
    
    output_path = os.path.join(project_root, "data", "brain_accuracy_report.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📂 大腦預測力審計報告已安全導出至：{output_path}")
    except Exception as e:
        print(f"⚠️ 大腦審計報告導出失敗: {e}")
        
    # 格式化輸出
    print("\n" + "="*50)
    print("📊 【AVM 大腦預測力獨立審計報告 (V4.0 Engine)】")
    print("="*50)
    print(f"  - 3日預測相關性 (IC_3d)  : {ic_3d:+.4f}")
    print(f"  - 5日預測相關性 (IC_5d)  : {ic_5d:+.4f}")
    print(f"  - 10日預測相關性 (IC_10d): {ic_10d:+.4f}")
    print(f"  - 高信心Directional Precision (p>=0.70): {directional_precision_70*100:.2f} %")
    print("="*50 + "\n")
    return report

if __name__ == "__main__":
    run_brain_accuracy_audit()
