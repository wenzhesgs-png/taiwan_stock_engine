import os
import sys
import json
from google import genai
from google.genai import types

# 🔑 讀取環境變數中的 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY if API_KEY else "dummy_key_for_init", http_options={'timeout': 10.0})

def run_ai_analysis(
    latest_features: dict,
    report_path: str = "data/llm_agent_report.json"
) -> dict:
    """
    執行 AI qualitative 戰術 analysis (TICKET-FIX-AI-MEM)
    - 記憶體直接注入：接收已計算好的特徵字典，無硬碟路徑相依。
    - 嚴格的 Fail-Fast 資料驗證門禁。
    - 支援 10s Timeout 與 Fallback 降級保護。
    """
    # 1. Fail-Fast 阻斷門禁
    if not latest_features:
        raise ValueError("最新特徵資料不能為空")
        
    date = latest_features.get("date")
    close = latest_features.get("close")
    kd_summary = latest_features.get("kd_summary")
    macd_summary = latest_features.get("macd_summary")
    
    if not date or str(date).strip() == "":
        raise ValueError("最新特徵資料中缺乏 'date' 欄位或日期為空值")
    
    if close is None:
        raise ValueError("最新特徵資料中缺乏 'close' 欄位")
        
    try:
        close_val = float(close)
    except (ValueError, TypeError):
        raise ValueError(f"收盤價格式不合規且無法轉換為浮點數: {close}")
        
    if close_val <= 0:
        raise ValueError(f"收盤價異常，必須為大於 0 的正數: {close_val}")
        
    if not kd_summary or str(kd_summary).strip() == "":
        raise ValueError("技術指標 'kd_summary' 欄位遺失或為空值")
        
    if not macd_summary or str(macd_summary).strip() == "":
        raise ValueError("技術指標 'macd_summary' 欄位遺失或為空值")

    # 2. 構造技術摘要
    kd_part = kd_summary.split('(')[1].split(')')[0] if '(' in str(kd_summary) else str(kd_summary)
    macd_part = macd_summary.split('(')[1].split(')')[0] if '(' in str(macd_summary) else str(macd_summary)
    tech_summary = f"KD: {kd_part.strip()} / MACD: {macd_part.strip()}"

    # 3. 準備保底降級回覆 (Fallback Response)
    fallback_res = {
        "date": str(date).strip(),
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

    # 4. 檢查 API Key 缺失
    current_key = API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not current_key or current_key == "":
        print("⚠️ [Warning] GEMINI_API_KEY 未設定或為空字串，將安全啟用技術面保底戰報。", file=sys.stderr)
        _write_report_file(fallback_res, report_path)
        return fallback_res

    # 5. 構建軍師王謀定性分析 Prompt
    system_instruction = (
        "你名叫「王謀」，是股市最倚重的首席戰術軍師。\n"
        "請根據主帥提供的盤後價格與技術面指標，執行極富進攻性的「五層思考定性分析」，並制定明天的交易作戰計畫。\n"
        "你必須嚴格遵守輸出 JSON 規格，不包含任何 Markdown 標記、JSON 標籤、或任何前後言雜訊。"
    )
    
    user_prompt = f"""
    主帥提供最新盤後情報：
    • 標的名稱/代碼：瑞昱 2379.TW
    • 數據截止日期：{date}
    • 今日收盤價：{close_val} 元
    • 技術指標狀態：KD - {kd_summary} / MACD - {macd_summary}
    
    請結合上述數據制定明天的事前交易觸發計畫：
    1. 評估「預估勝率 (win_rate, % 為 0 至 100 之間整數)」與「建議倉位 (suggested_position, 0.0 到 1.0 之間浮點數)」。
    2. 提供具體進場計畫 (entry_plan) 與出場計畫 (exit_plan)，格式必須包含具體目標價格（如 $XXX）以及相對今日收盤價 {close_val} 元的百分比變動（如 -X.X% 或 +X.X%）。
    3. 跳空防守 (gap_defense_note) 必須明確標註「若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價」之防守原則。
    4. 輸出限制：字數王謀分析在 200 字以內。
    
    必須嚴格以下列 JSON 格式直接回覆：
    {{
      "date": "{date}",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉) / 觀望 / 進場",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $XXX (-X.X%) 考慮進場 / 跌破 $XXX (-X.X%) 必買",
      "exit_plan": "達 $XXX (+X.X%) 停利 / 跌破 $XXX (-X.X%) 嚴格停損",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "軍師王謀定性分析（200字以內）"
    }}
    """

    # 6. 10 秒 Timeout 與 API 容錯
    try:
        response = client.models.generate_content(
            model='gemini-3.7-flash',
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
        
        # 7. JSON Schema 欄位缺失補防
        final_res = {
            "date": str(data.get("date", date)).strip(),
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
        print(f"[ERROR] Gemini API Failed: {type(e).__name__} - {e}", file=sys.stderr)
        _write_report_file(fallback_res, report_path)
        return fallback_res

def _write_report_file(res: dict, report_path: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 寫入報告 JSON 失敗 ({e})")
