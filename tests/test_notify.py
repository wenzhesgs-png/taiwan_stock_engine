import os
import json
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import URLError, HTTPError

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "src")))

from notify import send_telegram_notification


@pytest.fixture
def temp_report_file(tmp_path):
    """建立臨時的 llm_agent_report.json (新 JSON Schema)"""
    report_data = {
        "date": "2026-08-12",
        "close": 766.0,
        "technical_summary": "KD 低檔超賣 (K=15.2, D=18.5) / MACD 空頭收斂 (Hist=-2.1)",
        "win_rate": 65,
        "action": "小資金試單 (搶V轉)",
        "suggested_position": 0.1,
        "entry_plan": "回測至 $750 (-2.1%) 考慮進場 / 跌破 $740 (-3.4%) 必買",
        "exit_plan": "達 $800 (+4.4%) 停利 / 跌破 $730 (-4.7%) 嚴格停損",
        "gap_defense_note": "若開盤跳空開低於停損價或大幅跳空開高，本計畫失效，嚴禁盲目追價",
        "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，籌碼面主力法人大買，戰術定奪：小資金試單，跟隨獲利奔跑。"
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

    # 驗證傳送的訊息內容是否包含所有新欄位，且無 Debug 標記 (如 action, suggested_position, llm_fallback 鍵名括號)
    import urllib.parse
    req_body = urllib.parse.unquote_plus(req.data.decode("utf-8"))
    assert "2026-08-12" in req_body
    assert "766.0" in req_body
    assert "KD" in req_body
    assert "小資金試單 (搶V轉)" in req_body
    assert "65%" in req_body
    assert "10%" in req_body
    assert "750" in req_body
    assert "800" in req_body
    assert "跳空" in req_body
    assert "主力在多頭均線" in req_body
    assert "(action)" not in req_body
    assert "(suggested_position)" not in req_body
    assert "(llm_fallback)" not in req_body


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


@patch("urllib.request.urlopen")
def test_send_notification_legacy_json_fallback(mock_urlopen, tmp_path):
    """情境 6：載入不完整或舊版之 JSON 格式，測試預設值補全保護與不崩潰"""
    # 僅有部分舊版欄位
    legacy_data = {
        "action": "進場",
        "suggested_position": 0.5,
        "wang_mou_analysis": "舊版診斷內容"
    }
    report_file = tmp_path / "legacy_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f, indent=2, ensure_ascii=False)

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=str(report_file),
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    # 驗證預設值填補成功
    import urllib.parse
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "YYYY-MM-DD" in req_body
    assert "進場" in req_body
    assert "50%" in req_body
    assert "極端跳空" in req_body
