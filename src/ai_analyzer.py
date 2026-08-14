import os
import json
import pandas as pd
import numpy as np
import google.genai as genai
from google.genai import types

# 🔑 讀取環境變數中的 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY if API_KEY else "dummy_key_for_init", http_options={'timeout': 10.0})

def _parse_raw_price_data(raw_data_path: str) -> pd.DataFrame:
    """解析 yfinance 下載的 raw_2379.csv 數據"""
    if not os.path.exists(raw_data_path):
        return pd.DataFrame()
    try:
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
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_price.columns:
                df_price[col] = pd.to_numeric(df_price[col], errors='coerce')
        df_price = df_price.dropna(subset=['Close'])
        df_price['Date'] = pd.to_datetime(df_price['Date'], errors='coerce')
        df_price = df_price[df_price['Date'].notnull()].sort_values('Date')
        return df_price
    except Exception:
        return pd.DataFrame()

def compute_technical_summary(df_price: pd.DataFrame) -> tuple[float, str]:
    """
    從價格數據中計算最新收盤價及 KD / MACD 狀態摘要
    """
    if df_price.empty:
        return 0.0, "數據不足"
    
    latest_row = df_price.iloc[-1]
    close_val = round(float(latest_row['Close']), 2)
    
    # 計算 KD (9, 3, 3)
    try:
        df = df_price.copy()
        low_9 = df['Low'].rolling(window=9).min()
        high_9 = df['High'].rolling(window=9).max()
        df['RSV'] = np.where(high_9 != low_9, (df['Close'] - low_9) / (high_9 - low_9) * 100, 50)
        df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
        
        latest_k = round(float(df['K'].iloc[-1]), 1)
        latest_d = round(float(df['D'].iloc[-1]), 1)
        
        kd_status = "低檔超賣" if latest_k < 20 else ("高檔超買" if latest_k > 80 else "中性震盪")
        kd_summary = f"KD {kd_status} (K={latest_k}, D={latest_d})"
    except Exception:
        kd_summary = "KD 狀態未明"

    # 計算 MACD (12, 26, 9)
    try:
        close_series = df_price['Close']
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_val = ema12 - ema26
        signal_val = macd_val.ewm(span=9, adjust=False).mean()
        hist_val = macd_val - signal_val
        
        latest_macd = round(float(macd_val.iloc[-1]), 2)
        latest_sig = round(float(signal_val.iloc[-1]), 2)
        latest_hist = round(float(hist_val.iloc[-1]), 2)
        
        macd_status = "多頭增強" if latest_hist > 0 else "空頭收斂"
        macd_summary = f"MACD {macd_status} (Hist={latest_hist})"
    except Exception:
        macd_summary = "MACD 狀態未明"

    tech_summary = f"{kd_summary} / {macd_summary}"
    return close_val, tech_summary

def run_ai_analysis(
    raw_data_path: str = "data/raw_2379.csv",
    report_path: str = "data/llm_agent_report.json"
) -> dict:
    """
    執行 AI qualitative 戰術分析 (TICKET-FIX-AI)
    - 載入歷史數據，計算最新收盤價與 KD/MACD 指標
    - 調用 Gemini API 生成包含價格與百分比的隔日交易計畫
    - 支援 10s Timeout、JSON 補防與 Fallback 降級保護
    """
    df_price = _parse_raw_price_data(raw_data_path)
    
    close_val = 0.0
    tech_summary = "數據載入異常"
    latest_date = "YYYY-MM-DD"
    
    if not df_price.empty:
        close_val, tech_summary = compute_technical_summary(df_price)
        latest_date = df_price.iloc[-1]['Date'].strftime('%Y-%m-%d')
    
    # 1. 準備保底降級回覆 (Fallback Response)
    fallback_res = {
        "date": latest_date,
        "close": close_val,
        "technical_summary": tech_summary,
        "win_rate": 50,
        "action": "觀望",
        "suggested_position": 0.0,
        "entry_plan": f"回測至 {round(close_val * 0.97, 1)} (-3.0%) 考慮進場",
        "exit_plan": f"達 {round(close_val * 1.05, 1)} (+5.0%) 停利 / 跌破 {round(close_val * 0.95, 1)} (-5.0%) 嚴格停損",
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "API 連線異常，啟動防禦性降級保底，暫時維持觀望避開震盪。",
        "llm_fallback": True
    }

    # 2. 檢查 API Key 缺失
    current_key = API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not current_key or current_key == "":
        _write_report_file(fallback_res, report_path)
        return fallback_res

    # 3. 構建軍師王謀定性分析 Prompt
    system_instruction = (
        "你名叫「王謀」，是股市最倚重的首席戰術軍師。\n"
        "請根據主帥提供的盤後價格與技術面指標，執行極富進攻性的「五層思考定性分析」，並制定明天的交易作戰計畫。\n"
        "你必須嚴格遵守輸出 JSON 規格，不包含任何 Markdown 標記、JSON 標籤、或任何前後言雜訊。"
    )
    
    user_prompt = f"""
    主帥提供最新盤後情報：
    • 標的名稱/代碼：瑞昱 2379.TW
    • 數據截止日期：{latest_date}
    • 今日收盤價：{close_val} 元
    • 技術指標狀態：{tech_summary}
    
    請結合上述數據制定明天的事前交易觸發計畫：
    1. 評估「預估勝率 (win_rate, % 為 0 至 100 之間整數)」與「建議倉位 (suggested_position, 0.0 到 1.0 之間浮點數)」。
    2. 提供具體進場計畫 (entry_plan) 與出場計畫 (exit_plan)，格式必須包含具體目標價格（如 $XXX）以及相對今日收盤價 {close_val} 元的百分比變動（如 -X.X% 或 +X.X%）。
    3. 跳空防守 (gap_defense_note) 必須明確標註「若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價」之防守原則。
    4. 輸出限制：字數王謀分析在 200 字以內。
    
    必須嚴格以下列 JSON 格式直接回覆：
    {{
      "date": "{latest_date}",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉) / 觀望 / 進場",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $XXX (-X.X%) 考慮進場 / 跌破 $XXX (-X.X%) 必買",
      "exit_plan": "達 $XXX (+X.X%) 停利 / 跌破 $XXX (-X.X%) 嚴格停損",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "軍師王謀定性分析（200字以內）"
    }}
    """

    # 4. 10 秒 Timeout 與 API 容錯
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=user_prompt.strip(),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
        )
        
        text = response.text.strip()
        # 修剪 Markdown 標記
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        data = json.loads(text)
        
        # 5. JSON Schema 欄位缺失補防
        final_res = {
            "date": str(data.get("date", latest_date)).strip(),
            "close": close_val,
            "technical_summary": tech_summary,
            "win_rate": int(data.get("win_rate", 50)),
            "action": str(data.get("action", "觀望")).strip(),
            "suggested_position": float(data.get("suggested_position", 0.0)),
            "entry_plan": str(data.get("entry_plan", fallback_res["entry_plan"])).strip(),
            "exit_plan": str(data.get("exit_plan", fallback_res["exit_plan"])).strip(),
            "gap_defense_note": str(data.get("gap_defense_note", fallback_res["gap_defense_note"])).strip(),
            "wang_mou_analysis": str(data.get("wang_mou_analysis", "分析未明。")).strip(),
            "llm_fallback": False
        }
        
        # 限制字數與大小範圍
        if len(final_res["wang_mou_analysis"]) > 200:
            final_res["wang_mou_analysis"] = final_res["wang_mou_analysis"][:197] + "..."
        final_res["suggested_position"] = max(0.0, min(1.0, final_res["suggested_position"]))
        final_res["win_rate"] = max(0, min(100, final_res["win_rate"]))
        
        _write_report_file(final_res, report_path)
        return final_res

    except Exception as e:
        print(f"⚠️ [ai_analyzer] API 異常或解析失敗 ({e})，啟動防禦性降級。")
        _write_report_file(fallback_res, report_path)
        return fallback_res

def _write_report_file(res: dict, report_path: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 寫入報告 JSON 失敗 ({e})")
