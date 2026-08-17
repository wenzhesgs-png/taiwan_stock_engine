import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from src.ai_analyzer import run_ai_analysis

@pytest.fixture
def temp_report_file(tmp_path):
    return str(tmp_path / "llm_agent_report.json")

@pytest.fixture
def valid_features():
    return {
        "date": "2026-08-17",
        "close": 714.0,
        "volume": 12500000,
        "kd_summary": "K: 25.4, D: 28.1 (低檔鈍化 / 潛在黃金交叉)",
        "macd_summary": "DIF: -2.1, MACD: -1.8, OSC: -0.3 (綠柱縮腳)"
    }

def test_ai_analyzer_fail_fast_missing_all():
    """驗證 Fail-Fast：傳入特徵字典為空時拋出 ValueError"""
    with pytest.raises(ValueError, match="最新特徵資料不能為空"):
        run_ai_analysis(latest_features={})

def test_ai_analyzer_fail_fast_missing_date(valid_features):
    """驗證 Fail-Fast：缺乏 Date 或日期為空時拋出 ValueError"""
    features = valid_features.copy()
    features["date"] = ""
    with pytest.raises(ValueError, match="缺乏 'date' 欄位或日期為空值"):
        run_ai_analysis(latest_features=features)

def test_ai_analyzer_fail_fast_missing_close(valid_features):
    """驗證 Fail-Fast：缺乏 Close 欄位時拋出 ValueError"""
    features = valid_features.copy()
    features.pop("close")
    with pytest.raises(ValueError, match="缺乏 'close' 欄位"):
        run_ai_analysis(latest_features=features)

def test_ai_analyzer_fail_fast_invalid_close_format(valid_features):
    """驗證 Fail-Fast：收盤價格式不合規且無法轉換為浮點數時拋出 ValueError"""
    features = valid_features.copy()
    features["close"] = "invalid_price"
    with pytest.raises(ValueError, match="收盤價格式不合規且無法轉換為浮點數"):
        run_ai_analysis(latest_features=features)

def test_ai_analyzer_fail_fast_negative_close(valid_features):
    """驗證 Fail-Fast：收盤價為零或負數時拋出 ValueError"""
    features = valid_features.copy()
    features["close"] = -10.0
    with pytest.raises(ValueError, match="收盤價異常，必須為大於 0 的正數"):
        run_ai_analysis(latest_features=features)

def test_ai_analyzer_fail_fast_missing_kd(valid_features):
    """驗證 Fail-Fast：kd_summary 缺失或為空值時拋出 ValueError"""
    features = valid_features.copy()
    features["kd_summary"] = ""
    with pytest.raises(ValueError, match="技術指標 'kd_summary' 欄位遺失或為空值"):
        run_ai_analysis(latest_features=features)

def test_ai_analyzer_fail_fast_missing_macd(valid_features):
    """驗證 Fail-Fast：macd_summary 缺失或為空值時拋出 ValueError"""
    features = valid_features.copy()
    features["macd_summary"] = " "
    with pytest.raises(ValueError, match="技術指標 'macd_summary' 欄位遺失或為空值"):
        run_ai_analysis(latest_features=features)


@patch("src.ai_analyzer.genai.Client")
def test_ai_analyzer_success(mock_client_class, valid_features, temp_report_file):
    """情境 1：首選模型 (gemini-3.7-flash) 調用成功，返回合規之輸出"""
    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉)",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $692.6 (-3.0%) 考慮進場",
      "exit_plan": "達 $749.7 (+5.0%) 停利 / 跌破 $678.3 (-5.0%) 嚴格停損",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "主力多頭洗盤，指標高檔，王謀定性建議小資金摸底試單。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-17"
        assert res["close"] == 714.0
        assert res["win_rate"] == 65
        assert res["action"] == "小資金試單 (搶V轉)"
        assert res["suggested_position"] == 0.1
        assert "692.6" in res["entry_plan"]
        assert "749.7" in res["exit_plan"]
        assert "跳空" in res["gap_defense_note"]
        assert "王謀定性建議" in res["wang_mou_analysis"]
        assert res["llm_fallback"] is False
        assert res["technical_summary"] == "KD: 低檔鈍化 / 潛在黃金交叉 / MACD: 綠柱縮腳"
        
        # 驗證首次調用時傳入了 gemini-3.7-flash
        mock_client.models.generate_content.assert_called_once()
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs.get("model") == "gemini-3.7-flash"


@patch("src.ai_analyzer.genai.Client")
@patch("time.sleep")
def test_ai_analyzer_cascade_success(mock_sleep, mock_client_class, valid_features, temp_report_file, capsys):
    """情境 2：首選模型 (gemini-3.7-flash) 調用失敗，自動無縫降級切換至備援模型 (gemini-3.6-flash) 且成功"""
    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉)",
      "suggested_position": 0.1,
      "entry_plan": "回測至 692.6 (-3.0%) 考慮進場",
      "exit_plan": "達 749.7 (+5.0%) 停利 / 跌破 678.3 (-5.0%) 嚴格停損",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "主力多頭洗盤，指標高檔，王謀定性建議小資金摸底試單。"
    }
    """
    mock_client = MagicMock()
    # 第一回拋出 503 UNAVAILABLE 錯誤，第二回成功回傳
    mock_client.models.generate_content.side_effect = [
        Exception("503 UNAVAILABLE"),
        MagicMock(text=mock_response_text)
    ]
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-17"
        assert res["close"] == 714.0
        assert res["win_rate"] == 65
        assert res["llm_fallback"] is False
        
        # 驗證呼叫了 2 次 models.generate_content
        assert mock_client.models.generate_content.call_count == 2
        
        # 驗證模型依序為 gemini-3.7-flash 與 gemini-3.6-flash
        calls = mock_client.models.generate_content.call_args_list
        assert calls[0][1].get("model") == "gemini-3.7-flash"
        assert calls[1][1].get("model") == "gemini-3.6-flash"
        
        # 驗證 time.sleep(2) 觸發
        mock_sleep.assert_called_once_with(2)
        
        # 驗證 sys.stderr 輸出切換警告日誌
        captured = capsys.readouterr()
        assert "[Warning] Primary model (gemini-3.7-flash) failed: Exception - 503" in captured.err


@patch("src.ai_analyzer.genai.Client")
@patch("time.sleep")
def test_ai_analyzer_cascade_fail_throws_exception(mock_sleep, mock_client_class, valid_features, temp_report_file, capsys):
    """情境 3：主備模型皆失敗，Fail-Fast 直接拋出例外中斷且向 sys.stderr 輸出具體 [ERROR] 與 Traceback"""
    mock_client = MagicMock()
    # 兩次皆拋出錯誤
    mock_client.models.generate_content.side_effect = [
        Exception("Primary Timeout"),
        Exception("Fallback Overloaded 503")
    ]
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        with pytest.raises(Exception, match="Fallback Overloaded 503"):
            run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
            
        assert mock_client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once_with(2)
        
        # 驗證 sys.stderr 輸出正確的切換警告、[ERROR] 與 Traceback
        captured = capsys.readouterr()
        assert "[Warning] Primary model (gemini-3.7-flash) failed: Exception - Primary Timeout" in captured.err
        assert "[ERROR] Gemini API Failed on both primary and fallback models: Exception - Fallback Overloaded 503" in captured.err


def test_ai_analyzer_missing_api_key_throws_value_error(valid_features, temp_report_file, capsys):
    """情境 4：當 GEMINI_API_KEY 缺失時，立即拋出 ValueError("GEMINI_API_KEY 未設定") 並向 sys.stderr 輸出錯誤"""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with pytest.raises(ValueError, match="GEMINI_API_KEY 未設定"):
            run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
            
        captured = sys.stderr.getvalue() if hasattr(sys.stderr, "getvalue") else capsys.readouterr().err
        assert "[ERROR] GEMINI_API_KEY 未設定" in captured
