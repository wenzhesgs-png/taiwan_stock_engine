import os
import json
import urllib.request
import urllib.parse
from typing import Optional

def send_telegram_notification(
    report_path: str = "data/llm_agent_report.json",
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None
) -> bool:
    """
    將 AI 股市預測報告推播至 Telegram Channel 或 Chat。
    若憑證缺失、檔案不存在、逾時或連線異常，皆會優雅降級不崩潰，並回傳 False。
    """
    # 讀取環境變數作為預設值
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    
    # 憑證缺失防禦
    if not bot_token or not chat_id:
        print("⚠️ [Warning] Telegram 推播認證憑證缺失 (TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID 未設定)。將自動跳過。")
        return False

    # 報告檔案不存在防禦
    if not os.path.exists(report_path):
        print(f"⚠️ [Warning] 找不到指定的戰術報告檔案: '{report_path}'。將自動跳過。")
        return False

    try:
        # 讀取 JSON 報告
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        print(f"⚠️ [Warning] 讀取或解析戰術報告檔案時發生錯誤 ({e})。將自動跳過。")
        return False

    # 欄位缺失補防與解析
    date = report_data.get("date", "YYYY-MM-DD")
    close = report_data.get("close", 0.0)
    technical_summary = report_data.get("technical_summary", "技術指標數據不足")
    action = report_data.get("action", "觀望")
    win_rate = report_data.get("win_rate", 50)
    suggested_position = report_data.get("suggested_position", 0.0)
    entry_plan = report_data.get("entry_plan", "未設定")
    exit_plan = report_data.get("exit_plan", "未設定")
    gap_defense_note = report_data.get("gap_defense_note", "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價")
    wang_mou_analysis = report_data.get("wang_mou_analysis", "無分析內容")

    # 建議倉位轉換為百分比
    try:
        suggested_position_pct = float(suggested_position) * 100
        if suggested_position_pct.is_integer():
            suggested_position_pct = int(suggested_position_pct)
        else:
            suggested_position_pct = round(suggested_position_pct, 1)
    except Exception:
        suggested_position_pct = 0

    # 格式化為高資訊密度 Markdown 排版 (去除 Debug 噪訊與括號變數)
    message = (
        "🔔 【AI 股市決策戰報 - 瑞昱 2379.TW】\n"
        f"📅 數據截止日期：{date} 盤後\n\n"
        "📊 盤後關鍵指標：\n"
        f"• 收盤價：{close} 元\n"
        f"• KD / MACD 狀態：{technical_summary}\n\n"
        "🎯 軍師王謀戰術決策：\n"
        f"• 動作建議：{action}\n"
        f"• 預估勝率：{win_rate}%\n"
        f"• 建議倉位：{suggested_position_pct}%\n\n"
        "📝 隔日事前觸發計畫：\n"
        f"• 🟢 進場條件：{entry_plan}\n"
        f"• 🔴 出場條件：{exit_plan}\n"
        f"• ⚠️ 跳空防守：{gap_defense_note}\n\n"
        "🧠 軍師王謀定性診斷：\n"
        f"{wang_mou_analysis}"
    )

    # 建構 Telegram API 呼叫 URL
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        
        # 硬性設定 5 秒 Timeout
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            if status_code == 200:
                print("✨ [Success] Telegram 戰術戰報推播成功！")
                return True
            else:
                print(f"⚠️ [Warning] Telegram 推播 API 回傳非 200 狀態碼: {status_code}。將自動跳過。")
                return False
    except Exception as e:
        print(f"⚠️ [Warning] Telegram 推播異常 (連線逾時、網路錯誤或 API 回傳錯誤): {e}。將優雅跳過。")
        return False
