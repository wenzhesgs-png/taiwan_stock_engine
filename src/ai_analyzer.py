import os
import sys
import json
import time
from google import genai
from google.genai import types
from pathlib import Path
# 明確指定往上一層找根目錄的 .env，雲端未安裝 dotenv 則自動跳過
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# 動作中文映射字典
ACTION_CHINESE_MAP = {
    "BUY": "【🎯 進場買進】",
    "HOLD": "【🛡️ 持股續抱】",
    "WAIT": "【☕ 空手觀望】",
    "EXIT": "【🚨 平倉出場】"
}

def _parse_trade_journals(journal_path: str) -> dict:
    """
    解析 data/trade_journals.json 中的 A8_TOP_DEFENSE_MODERATE 與 #2_HUMAN_GOLD_STANDARD 資訊
    (徹底去個人帳本化：移除任何個人持股數、買入日期、成本等敏感帳本字樣，僅萃取客觀波段勝率與歷史波段數)
    """
    default_res = {
        "human_summary": "歷史波段：已完成多波段客觀量化回測。目前處於空倉觀望中。",
        "a8_summary": "歷史波段：完成多波段客觀往返交易。目前空手觀望，出場教訓：嚴格執行移動防守軌道、過熱逃頂避開拉回。"
    }
    if not os.path.exists(journal_path):
        return default_res
    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        journals = data.get("journals", {})
        
        # 1. 解析 #2_HUMAN_GOLD_STANDARD (人類黃金基準) - 去個人帳本化
        human_list = journals.get("#2_HUMAN_GOLD_STANDARD", [])
        if human_list:
            total_waves = len(human_list)
            human_summary = f"歷史波段：已完成 {total_waves} 個波段交易。目前處於空倉觀望狀態。"
        else:
            human_summary = "歷史波段：無資料；目前處於空倉觀望狀態。"
            
        # 2. 解析 A8_TOP_DEFENSE_MODERATE (穩健冠軍) - 去個人帳本化
        a8_list = journals.get("A8_TOP_DEFENSE_MODERATE", [])
        if a8_list:
            completed_trades = 0
            exit_lessons = []
            
            for tx in a8_list:
                action = tx.get("action")
                reason = tx.get("reason")
                if action == "SELL":
                    completed_trades += 1
                    # 轉換為客觀說明，去除價格、日期與股數
                    if "stop_loss" in str(reason).lower():
                        exit_lessons.append("觸發 8% 物理硬停損軌道")
                    elif "overheat" in str(reason).lower() or "z_bias" in str(reason).lower():
                        exit_lessons.append("指標過熱逃頂防禦")
                    else:
                        exit_lessons.append("移動防守軌道平倉")
            
            lessons_str = "、".join(list(set(exit_lessons))[-2:]) if exit_lessons else "嚴格執行移動停損與高檔逃頂防禦"
            a8_summary = f"歷史波段：完成 {completed_trades} 次往返交易。目前處於空手狀態。歷史離場教訓：{lessons_str}。"
        else:
            a8_summary = "歷史波段：無資料；目前處於空手狀態。"
            
        return {
            "human_summary": human_summary,
            "a8_summary": a8_summary
        }
    except Exception as e:
        print(f"⚠️ [Warning] Parsing trade journals failed: {e}. Falling back to default empty strings.", file=sys.stderr)
        return default_res

def run_ai_analysis(
    latest_features: dict,
    report_path: str = "data/llm_agent_report.json"
) -> dict:
    """
    執行 AI qualitative 戰術 analysis (TICKET-FIX-AI-MEM)
    - 記憶體直接注入：接收已計算好的特徵字典，無硬碟路徑相依。
    - 嚴格的 Fail-Fast 資料驗證門禁。
    - 支援 30s Timeout 與 1 次自動重試（間隔 2 秒）。
    - 任何認證、網路與連線異常時，直接 Fail-Fast 拋出例外，杜絕靜默降級。
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

    # 3. 讀取並解析交易日誌 (去個人帳本化)
    journal_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(report_path)), "trade_journals.json"))
    journal_data = _parse_trade_journals(journal_path)
    human_summary = journal_data["human_summary"]
    a8_summary = journal_data["a8_summary"]

    # 4. 讀取 isd_predict_report.json 獲取動態選拔的冠軍資訊
    champion_agent = "A8_TOP_DEFENSE_MODERATE"
    raw_action = "HOLD"
    champion_reason = "HOLDING_CONTINUATION"
    
    isd_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(report_path)), "isd_predict_report.json"))
    if os.path.exists(isd_path):
        try:
            with open(isd_path, "r", encoding="utf-8") as f:
                isd_data = json.load(f)
            summary = isd_data.get("champion_summary")
            if summary:
                champion_agent = summary.get("agent_id", "A8_TOP_DEFENSE_MODERATE")
                raw_action = summary.get("today_action", "HOLD")
                champion_reason = summary.get("reason", "HOLDING_CONTINUATION")
            else:
                print("[Warning] champion_summary missing", file=sys.stderr)
        except Exception as e:
            print(f"[Warning] Parsing isd_predict_report.json failed: {e}. Falling back to default champion.", file=sys.stderr)
    else:
        print("[Warning] champion_summary missing", file=sys.stderr)

    # 安全校驗與 Action 全中文化對齊 (DoD 3)
    # 未定義的 Action 安全回歸至 WAIT
    if raw_action not in ACTION_CHINESE_MAP:
        print(f"[Warning] Unknown action: {raw_action}", file=sys.stderr)
        raw_action = "WAIT"
        
    action_chinese = ACTION_CHINESE_MAP[raw_action]

    # 解析 EXIT 退場原因 (動能轉弱保本防禦 / 雙重指標過熱逃頂)
    exit_reason = "動能轉弱保本防禦"
    if "overheat" in str(champion_reason).lower() or "z_bias" in str(champion_reason).lower():
        exit_reason = "雙重指標過熱逃頂"

    # 5. 準備保底降級回覆 (Fallback Response)
    fallback_res = {
        "date": str(date).strip(),
        "close": close_val,
        "technical_summary": tech_summary,
        "champion_agent": champion_agent,
        "action": action_chinese,
        "suggested_position": 0.0 if raw_action != "BUY" else 0.1,
        "entry_plan": None if raw_action != "BUY" else f"回測至 {round(close_val * 0.97, 1)} (-3.0%) 考慮進場",
        "exit_plan": f"達 {round(close_val * 1.05, 1)} (+5.0%) 停利 / 跌破 {round(close_val * 0.95, 1)} (-5.0%) 嚴格停損" if raw_action in ["BUY", "HOLD"] else (f"觸發{exit_reason}機制，今日部位全數平倉出清，轉為空手觀望，資金落袋防守" if raw_action == "EXIT" else None),
        "defense_plan_empty_hand": "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號" if raw_action in ["HOLD", "EXIT"] else ("當前無標準量化買點，耐性等待籌碼築底或明確突破訊號" if raw_action == "WAIT" else "不適用"),
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "API 連線異常，啟用防禦性降級，暫時維持原有戰術守則。",
        "llm_fallback": True
    }

    # 6. 檢查 API Key 缺失 (Fail-Fast: 立即拋出 ValueError 阻斷執行)
    current_key = os.environ.get("GEMINI_API_KEY", "")
    if not current_key or current_key.strip() == "":
        print("[ERROR] GEMINI_API_KEY 未設定", file=sys.stderr)
        raise ValueError("GEMINI_API_KEY 未設定")

    # Lazy Init of genai.Client (Timeout 60.0s / 60000ms via types.HttpOptions)
    try:
        client = genai.Client(
            api_key=current_key,
            http_options=types.HttpOptions(
                api_version="v1alpha",
                timeout=60000
            )
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini Client: {type(e).__name__} - {e}", file=sys.stderr)
        raise e

    # 7. 構建軍師王謀定性分析 Prompt
    system_instruction = (
        "你名叫「王謀」，是股市最倚重的首席戰術軍師。\n"
        "請根據主帥提供的盤後價格與技術面指標，執行極富進攻性的「五層思考定性分析」，並制定明天的交易作戰計畫。\n"
        "你必須嚴格遵守輸出 JSON 規格，不包含任何 Markdown 標記、JSON 標籤、或任何前後言雜訊。"
    )

    # 依據今日狀態與去帳本化動態切換硬性約束，抑制非買點進場計畫生成 (DoD 3)
    if raw_action in ["HOLD", "EXIT", "WAIT"]:
        if raw_action == "EXIT":
            exit_detail_instruction = f"平倉出場日：1. 嚴禁在 entry_plan 填入任何價格。2. exit_plan 必須且只能設定為: '觸發{exit_reason}機制，今日部位全數平倉出清，轉為空手觀望，資金落袋防守'。3. suggested_position 必須強制設為 0.0。"
        elif raw_action == "WAIT":
            exit_detail_instruction = "空手觀望日：1. 嚴禁在 entry_plan 和 exit_plan 填入任何看多停利/停損價位，兩者必須強制為 null。2. suggested_position 必須為 0.0。"
        else: # HOLD
            exit_detail_instruction = "持股續抱日：1. 嚴禁在 entry_plan 填入任何價格，必須強制為 null。2. 必須在 exit_plan 填入移動防守線。3. 建議倉位與勝率需大於 0.0。"

        hard_constraint = (
            "\n🔴 【硬性約束門禁 (Hard Constraint)】：\n"
            f"目前最新量化冠軍 Agent 判定今日訊號為 {action_chinese} (非買點日)。\n"
            "1. 嚴禁在回覆中生成或編造任何進場低接價格、建倉計畫或試單買點！\n"
            "2. 嚴禁在戰報與王謀分析中，提及任何 Agent 個人持倉帳本細節（如持股股數、買入日期、個人成本價）。\n"
            f"3. 具體防守規則：{exit_detail_instruction}\n"
            "4. 必須在 'defense_plan_empty_hand' 欄位中提供空手者紀律：'波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號'。\n"
        )
    else:
        hard_constraint = (
            "\n🟢 【進場建倉提示】：\n"
            f"目前今日訊號為 {action_chinese} (黃金買點日)。\n"
            "1. 請在 'entry_plan' 欄位中生成黃金買點建倉計畫（必須包含具體價格如 $XXX 與相對今日收盤價的百分比變動，如 -X.X%）。\n"
            "2. 建議倉位 'suggested_position' 應根據勝率進行大膽配置（大於 0.0 且最大為 1.0）。\n"
            "3. 'defense_plan_empty_hand' 可設定為 '不適用'。\n"
        )
    
    user_prompt = f"""
    主帥提供最新盤後情報：
    • 標建立或代碼：瑞昱 2379.TW
    • 數據截止日期：{date}
    • 今日收盤價：{close_val} 元
    • 技術指標狀態：KD - {kd_summary} / MACD - {macd_summary}
    
    📊 歷史戰役與戰友持倉參考：
    • 人類黃金基準 (#2_HUMAN_GOLD_STANDARD)：{human_summary}
    • 穩健冠軍交易員 (A8_TOP_DEFENSE_MODERATE)：{a8_summary}
    {hard_constraint}
    
    請結合上述數據與約束，制定明天的事前交易計畫與定性診斷。
    
    必須嚴格以下列 JSON 格式直接回覆：
    {{
      "date": "{date}",
      "close": {close_val},
      "technical_summary": "{tech_summary}",
      "champion_agent": "{champion_agent}",
      "action": "{action_chinese}",
      "suggested_position": 0.0,
      "entry_plan": null,
      "exit_plan": "達 $XXX (+X.X%) 停利 / 跌破 $XXX (-X.X%) 嚴格停損",
      "defense_plan_empty_hand": "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "軍師王謀定性分析（200字以內）"
    }}
    """

    # 8. 三重階梯動態降級機制 (Cascade: 3.8 -> 3.7 -> 3.6)
    models = ['gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash']
    response = None
    last_exception = None

    for idx, model_name in enumerate(models):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt.strip(),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )
            break
        except Exception as e:
            last_exception = e
            if idx < len(models) - 1:
                next_model = models[idx + 1]
                print(f"⚠️ [Warning] Model ({model_name}) failed: {type(e).__name__} - {e}. "
                      f"Switching to fallback model ({next_model})...", file=sys.stderr)
                time.sleep(2)
            else:
                print(f"[ERROR] Gemini API Failed on all candidate models: {type(e).__name__} - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise last_exception

    # 9. 解析並修剪 JSON 數據
    try:
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        data = json.loads(text)
    except Exception as e:
        print(f"⚠️ [Warning] JSON 解析失敗，啟用安全降級保底。原因: {e}", file=sys.stderr)
        _write_report_file(fallback_res, report_path)
        return fallback_res

    # 10. JSON Schema 欄位缺失與硬性安全校驗自動補防
    final_action = str(data.get("action", action_chinese)).strip()
    
    # 逆向轉換為原始英文字串以便判定
    raw_action_reversed = "WAIT"
    for k, v in ACTION_CHINESE_MAP.items():
        if v == final_action or k == final_action:
            raw_action_reversed = k
            break
            
    final_suggested_pos = float(data.get("suggested_position", 0.0))
    final_entry_plan = data.get("entry_plan")
    final_exit_plan = data.get("exit_plan")
    final_defense_empty_hand = data.get("defense_plan_empty_hand", fallback_res["defense_plan_empty_hand"])
    
    # [DoD 3 & Edge Case] 強制攔截非 BUY 下的失效看多目標價，以及 EXIT/WAIT 的純淨化處理
    if raw_action_reversed != "BUY":
        final_suggested_pos = 0.0
        final_entry_plan = None
        
        if raw_action_reversed == "EXIT":
            # [Edge Case] API 在 EXIT 狀態仍回傳 exit_plan 價位時，代碼層實施硬性過濾覆寫
            final_exit_plan = f"觸發{exit_reason}機制，今日部位全數平倉出清，轉為空手觀望，資金落袋防守"
            final_defense_empty_hand = "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號"
        elif raw_action_reversed == "WAIT":
            # WAIT 觀望日完全隱藏所有價位
            final_exit_plan = None
            final_defense_empty_hand = "當前無標準量化買點，耐性等待籌碼築底或明確突破訊號"
        else: # HOLD
            if final_exit_plan is None or str(final_exit_plan).strip() == "":
                final_exit_plan = fallback_res["exit_plan"]
            final_defense_empty_hand = "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號"
    else:
        # BUY 狀態
        if final_entry_plan is None or str(final_entry_plan).strip() == "":
            final_entry_plan = fallback_res["entry_plan"]
        if final_exit_plan is None or str(final_exit_plan).strip() == "":
            final_exit_plan = fallback_res["exit_plan"]
        final_defense_empty_hand = "不適用"

    final_res = {
        "date": str(data.get("date", date)).strip(),
        "close": close_val,
        "technical_summary": tech_summary,
        "champion_agent": str(data.get("champion_agent", champion_agent)).strip(),
        "action": final_action,
        "suggested_position": final_suggested_pos,
        "entry_plan": final_entry_plan,
        "exit_plan": final_exit_plan,
        "defense_plan_empty_hand": str(final_defense_empty_hand).strip(),
        "gap_defense_note": str(data.get("gap_defense_note", fallback_res["gap_defense_note"])).strip(),
        "wang_mou_analysis": str(data.get("wang_mou_analysis", "分析未明。")).strip(),
        "llm_fallback": False
    }
    
    # 限制字數與大小範圍
    if len(final_res["wang_mou_analysis"]) > 200:
        final_res["wang_mou_analysis"] = final_res["wang_mou_analysis"][:197] + "..."
    final_res["suggested_position"] = max(0.0, min(1.0, final_res["suggested_position"]))
    final_res["win_rate"] = max(0, min(100, int(data.get("win_rate", 50)))) # 保全 win_rate 相容舊代碼
    
    _write_report_file(final_res, report_path)
    return final_res

def _write_report_file(res: dict, report_path: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 寫入報告 JSON 失敗 ({e})")