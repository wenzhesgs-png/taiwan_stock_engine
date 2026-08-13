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

    action = report_data.get("action", "無")
    suggested_position = report_data.get("suggested_position", 0.0)
    wang_mou_analysis = report_data.get("wang_mou_analysis", "無分析內容")
    llm_fallback = report_data.get("llm_fallback", False)

    # 格式化為簡潔 Markdown
    message = (
        "🔔 *AI 股市預測與決策引擎 - 每日定性戰術報告*\n\n"
        f"🎯 *決策動作 (action)*: {action}\n"
        f"📈 *建議倉位 (suggested_position)*: {suggested_position}\n"
        f"🧠 *軍師王謀定性診斷 (wang_mou_analysis)*: {wang_mou_analysis}\n"
        f"🛡️ *LLM 降級狀態 (llm_fallback)*: {llm_fallback}"
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
                print("✨ [Success] Telegram 戰術報告推播成功！")
                return True
            else:
                print(f"⚠️ [Warning] Telegram 推播 API 回傳非 200 狀態碼: {status_code}。將自動跳過。")
                return False
    except Exception as e:
        print(f"⚠️ [Warning] Telegram 推播異常 (連線逾時、網路錯誤或 API 回傳錯誤): {e}。將優雅跳過。")
        return False
