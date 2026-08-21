import os
import json
import pytest
import urllib.parse
from unittest.mock import patch, MagicMock
from urllib.error import URLError, HTTPError

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "src")))

from notify import send_telegram_notification


@pytest.fixture
def temp_report_file_buy(tmp_path):
    """建立臨時的 llm_agent_report.json (BUY 狀態)"""
    report_data = {
        "date": "2026-08-18",
        "close": 738.0,
        "technical_summary": "KD: 中性震盪 / MACD: 空頭收斂",
        "champion_agent": "A8_TOP_DEFENSE_MODERATE",
        "action": "【🎯 進場買進】",
        "suggested_position": 0.1,
        "entry_plan": "回測至 692.6 (-3.0%) 考慮進場",
        "exit_plan": "達 $775.0 (+5.0%) 執行獲利了結 / 跌破 $715.0 (-3.1%) 嚴格停損撤退",
        "defense_plan_empty_hand": "不適用",
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，籌碼面主力法人大買，戰術定奪：小資金試單，跟隨獲利奔跑。"
    }
    report_file = tmp_path / "llm_agent_report_buy.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return str(report_file)


@pytest.fixture
def temp_report_file_hold(tmp_path):
    """建立臨時的 llm_agent_report.json (HOLD 狀態)"""
    report_data = {
        "date": "2026-08-18",
        "close": 738.0,
        "technical_summary": "KD: 中性震盪 / MACD: 空頭收斂",
        "champion_agent": "A8_TOP_DEFENSE_MODERATE",
        "action": "【🛡️ 持股續抱】",
        "suggested_position": 0.0,
        "entry_plan": None,
        "exit_plan": "達 $775.0 (+5.0%) 執行獲利了結 / 跌破 $715.0 (-3.1%) 嚴格停損撤退",
        "defense_plan_empty_hand": "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號",
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，籌碼面主力法人大買，戰術定奪：持股者繼續波段抱緊。"
    }
    report_file = tmp_path / "llm_agent_report_hold.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return str(report_file)


@pytest.fixture
def temp_report_file_hold_with_new_buy(tmp_path):
    """建立臨時的 llm_agent_report.json (HOLD 且觸發新買點狀態)"""
    report_data = {
        "date": "2026-08-18",
        "close": 738.0,
        "technical_summary": "KD: 中性震盪 / MACD: 空頭收斂",
        "champion_agent": "A8_TOP_DEFENSE_MODERATE",
        "action": "【🛡️ 持股續抱】",
        "today_new_buy_triggered": True,
        "suggested_position": 0.1,
        "entry_plan": "回測至 692.6 (-3.0%) 考慮進場",
        "exit_plan": "達 $775.0 (+5.0%) 執行獲利了結 / 跌破 $715.0 (-3.1%) 嚴格停損撤退",
        "defense_plan_empty_hand": "不適用",
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "技術面再度觸發黃金買點，允許空手者次級上車！"
    }
    report_file = tmp_path / "llm_agent_report_hold_new_buy.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return str(report_file)


@pytest.fixture
def temp_report_file_wait(tmp_path):
    """建立臨時的 llm_agent_report.json (WAIT 狀態)"""
    report_data = {
        "date": "2026-08-18",
        "close": 738.0,
        "technical_summary": "KD: 中性震盪 / MACD: 空頭收斂",
        "champion_agent": "A8_TOP_DEFENSE_MODERATE",
        "action": "【☕ 空手觀望】",
        "suggested_position": 0.0,
        "entry_plan": None,
        "exit_plan": None,
        "defense_plan_empty_hand": "當前無標準量化買點，耐性等待籌碼築底或明確突破訊號",
        "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
        "wang_mou_analysis": "多空未明，靜待下一輪訊號。"
    }
    report_file = tmp_path / "llm_agent_report_wait.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return str(report_file)


@patch("urllib.request.urlopen")
def test_send_notification_buy(mock_urlopen, temp_report_file_buy):
    """測試情境 1：BUY 狀態下，完整呈現進場條件與黃金買點計畫"""
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=temp_report_file_buy,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    
    # 驗證包含了中文化動作與黃金買點進場條件
    assert "【🎯 進場買進】" in req_body
    assert "🟢 進場條件" in req_body
    assert "692.6" in req_body
    assert "10%" in req_body
    assert "(action)" not in req_body


@patch("urllib.request.urlopen")
def test_send_notification_hold_with_new_buy(mock_urlopen, temp_report_file_hold_with_new_buy):
    """測試情境 2：HOLD 且觸發新買點狀態下，動作抬頭動態切換，並列渲染防守線與進場條件"""
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=temp_report_file_hold_with_new_buy,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    
    # 驗證動作抬頭動態切換
    assert "【🛡️ 持股續抱】(🔥 今日觸發空手者/加碼買點)" in req_body
    # 驗證並列渲染
    assert "🛡️ 持股者移動防守線" in req_body
    assert "🟢 空手者/加碼進場條件" in req_body
    assert "692.6" in req_body
    assert "10%" in req_body


@patch("urllib.request.urlopen")
def test_send_notification_hold(mock_urlopen, temp_report_file_hold):
    """測試情境 3：HOLD 狀態下，100% 隱藏進場條件，顯示持股防守與空手者紀律"""
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=temp_report_file_hold,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    
    # 驗證中文化動作抬頭與 100% 隱藏了進場條件
    assert "【🛡️ 持股續抱】" in req_body
    assert "🟢 進場條件" not in req_body
    assert "黃金買點建倉計畫" not in req_body
    
    # 驗證顯示移動防守與空手者紀律
    assert "🛡️ 持股者移動防守條件" in req_body
    assert "⏳ 空手者紀律" in req_body
    assert "0%" in req_body


@patch("urllib.request.urlopen")
def test_send_notification_wait(mock_urlopen, temp_report_file_wait):
    """測試情境 4：WAIT 狀態下，100% 隱藏進場條件與停損停利，顯示空倉觀望"""
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"ok": true, "result": {}}'
    mock_urlopen.return_value = mock_response

    success = send_telegram_notification(
        report_path=temp_report_file_wait,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is True
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    
    # 驗證中文化動作抬頭與 100% 隱藏了進場條件和停利停損價
    assert "【☕ 空手觀望】" in req_body
    assert "🟢 進場條件" not in req_body
    assert "停利" not in req_body
    assert "停損" not in req_body
    
    # 驗證顯示空倉觀望
    assert "☕ 觀望等待" in req_body


def test_send_notification_missing_credentials(temp_report_file_buy):
    """測試情境 5：憑證缺失（未設定 Token 或 Chat ID）"""
    success = send_telegram_notification(
        report_path=temp_report_file_buy,
        bot_token=None,
        chat_id=None
    )
    assert success is False


def test_send_notification_file_not_found():
    """測試情境 6：報告檔案不存在，應優雅跳過並回傳 False"""
    success = send_telegram_notification(
        report_path="non_existent_file.json",
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )
    assert success is False


@patch("urllib.request.urlopen")
def test_send_notification_timeout(mock_urlopen, temp_report_file_buy):
    """測試情境 7：API 請求逾時（超過 5 秒），應捕捉 Timeout 異常，回傳 False 且不崩潰"""
    mock_urlopen.side_effect = URLError("timeout")

    success = send_telegram_notification(
        report_path=temp_report_file_buy,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is False


@patch("urllib.request.urlopen")
def test_send_notification_http_error(mock_urlopen, temp_report_file_buy):
    """測試情境 8：API 回傳 4xx/5xx HTTP 錯誤，應捕捉異常，回傳 False 且不崩潰"""
    mock_urlopen.side_effect = HTTPError(
        url="https://api.telegram.org/botmock_bot_token/sendMessage",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=None
    )

    success = send_telegram_notification(
        report_path=temp_report_file_buy,
        bot_token="mock_bot_token",
        chat_id="mock_chat_id"
    )

    assert success is False


@patch("urllib.request.urlopen")
def test_send_notification_legacy_json_fallback(mock_urlopen, tmp_path):
    """測試情境 9：載入不完整或舊版之 JSON 格式，測試預設值補全保護與不崩潰"""
    legacy_data = {
        "action": "【🛡️ 持股續抱】",
        "suggested_position": 0.0,
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
    req_body = urllib.parse.unquote_plus(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "YYYY-MM-DD" in req_body
    assert "【🛡️ 持股續抱】" in req_body
    assert "0%" in req_body
    assert "極端跳空" in req_body
