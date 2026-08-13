import os
import json
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import URLError, HTTPError

# 匯入待測試的 notify 模組
# 由於此時 src/notify.py 可能尚未實作或不完整，這符合 TDD 的 Red 步驟。
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "src")))

from notify import send_telegram_notification


@pytest.fixture
def temp_report_file(tmp_path):
    """建立臨時的 llm_agent_report.json"""
    report_data = {
        "action": "進場",
        "suggested_position": 1.0,
        "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，籌碼面主力法人大買，戰術定奪：All-In滿倉，跟隨獲利奔跑。",
        "llm_fallback": False
    }
    report_file = tmp_path / "llm_agent_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return str(report_file)


@patch("urllib.request.urlopen")
def test_send_notification_success(mock_urlopen, temp_report_file):
    """情境 1：憑證齊全、檔案存在，Telegram API 回傳成功 (200 OK)"""
    # 模擬 urlopen 回傳 200
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    # 驗證 urlopen 被呼叫，且帶有正確的 URL
    mock_urlopen.assert_called_once()
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.full_url == "https://api.telegram.org/botmock_bot_token/sendMessage"
    assert req.get_method() == "POST"
    assert kwargs.get("timeout") == 5


def test_send_notification_missing_credentials(temp_report_file):
    """情境 2：憑證缺失（未設定 Token 或 Chat ID）"""
    # 1. 兩者皆缺失
    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token=None,
        chat_id=None
    )
    assert success is False

    # 2. 缺失 Bot Token
    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token=None,
        chat_id="mock_chat_id"
    )
    assert success is False

    # 3. 缺失 Chat ID
    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token="mock_bot_token",
        chat_id=None
    )
    assert success is False


def test_send_notification_file_not_found():
    """情境 3：報告檔案不存在，應優雅跳過並回傳 False"""
    success = send_telegram_notification(
        report_path="non_existent_file.json",
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )
    assert success is False


@patch("urllib.request.urlopen")
def test_send_notification_timeout(mock_urlopen, temp_report_file):
    """情境 4：API 請求逾時（超過 5 秒），應捕捉 Timeout 異常，回傳 False 且不崩潰"""
    # 模擬 urlopen 丟出 URLError (例如：timeout)
    mock_urlopen.side_effect = URLError("timeout")

    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is False


@patch("urllib.request.urlopen")
def test_send_notification_http_error(mock_urlopen, temp_report_file):
    """情境 5：API 回傳 4xx/5xx HTTP 錯誤，應捕捉異常，回傳 False 且不崩潰"""
    # 模擬 urlopen 丟出 HTTPError
    mock_urlopen.side_effect = HTTPError(
        url="https://api.telegram.org/botmock_bot_token/sendMessage",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=None
    )

    success = send_telegram_notification(
        report_path=temp_report_file,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is False
