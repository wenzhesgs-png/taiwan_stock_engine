import os
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
        "date": "2026-08-14",
        "close": 530.0,
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
    """情境 1：API 調用成功，返回合規之輸出"""
    mock_response_text = """
    {
      "date": "2026-08-14",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉)",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $515 (-2.8%) 考慮試單進場",
      "exit_plan": "達 $560 (+5.6%) 停利 / 跌破 $500 (-5.6%) 嚴格停損",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "主力多頭洗盤，指標高檔，王謀定性建議小資金摸底試單。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-14"
        assert res["close"] == 530.0
        assert res["win_rate"] == 65
        assert res["action"] == "小資金試單 (搶V轉)"
        assert res["suggested_position"] == 0.1
        assert "515" in res["entry_plan"]
        assert "560" in res["exit_plan"]
        assert "跳空" in res["gap_defense_note"]
        assert "王謀定性建議" in res["wang_mou_analysis"]
        assert res["llm_fallback"] is False
        # 驗證 technical_summary 的括號內容提取與組合
        assert res["technical_summary"] == "KD: 低檔鈍化 / 潛在黃金交叉 / MACD: 綠柱縮腳"


@patch("src.ai_analyzer.genai.Client")
def test_ai_analyzer_api_error_fallback(mock_client_class, valid_features, temp_report_file, capsys):
    """情境 2：API 逾時或報錯時，測試 Fallback 且向 sys.stderr 輸出結構化錯誤日誌"""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Connection Timeout")
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-14"  # 真實日期
        assert res["close"] == 530.0         # 真實最新收盤價
        assert "KD: 低檔鈍化 / 潛在黃金交叉" in res["technical_summary"] # 真實指標
        assert res["win_rate"] == 50
        assert res["action"] == "觀望"
        assert res["suggested_position"] == 0.0
        assert "514.1" in res["entry_plan"]  # 基於真實收盤價 530 * 0.97 計算
        assert "556.5" in res["exit_plan"]   # 基於真實收盤價 530 * 1.05 計算
        assert "防禦性降級" in res["wang_mou_analysis"]
        assert res["llm_fallback"] is True
        
        # 驗證 sys.stderr 輸出
        captured = capsys.readouterr()
        assert "[ERROR] Gemini API Failed: Exception - API Connection Timeout" in captured.err


def test_ai_analyzer_missing_api_key_fallback(valid_features, temp_report_file, capsys):
    """測試當 GEMINI_API_KEY 缺失時，安全回傳技術面保底戰報且向 sys.stderr 輸出警告"""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        assert res["date"] == "2026-08-14"
        assert res["close"] == 530.0
        assert res["llm_fallback"] is True
        
        # 驗證 sys.stderr 輸出警告
        captured = capsys.readouterr()
        assert "[Warning] GEMINI_API_KEY 未設定或為空字串" in captured.err
