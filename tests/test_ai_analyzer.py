import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from src.ai_analyzer import run_ai_analysis, _parse_trade_journals

@pytest.fixture
def temp_report_file(tmp_path):
    return str(tmp_path / "llm_agent_report.json")

@pytest.fixture
def mock_isd_report_buy(tmp_path):
    """建立包含 BUY 冠軍的 isd_predict_report.json"""
    isd_data = {
        "isdTriggered": False,
        "champion_summary": {
            "agent_id": "A8_TOP_DEFENSE_MODERATE",
            "annual_roi": 135.45,
            "today_action": "BUY",
            "holding_status": "EMPTY",
            "entry_price": 0.0,
            "shares": 0
        }
    }
    isd_file = tmp_path / "isd_predict_report.json"
    with open(isd_file, "w", encoding="utf-8") as f:
        json.dump(isd_data, f, indent=2, ensure_ascii=False)
    return str(isd_file)

@pytest.fixture
def mock_isd_report_hold(tmp_path):
    """建立包含 HOLD 冠軍的 isd_predict_report.json"""
    isd_data = {
        "isdTriggered": False,
        "champion_summary": {
            "agent_id": "A8_TOP_DEFENSE_MODERATE",
            "annual_roi": 135.45,
            "today_action": "HOLD",
            "holding_status": "HOLDING",
            "entry_price": 700.76,
            "shares": 305
        }
    }
    isd_file = tmp_path / "isd_predict_report.json"
    with open(isd_file, "w", encoding="utf-8") as f:
        json.dump(isd_data, f, indent=2, ensure_ascii=False)
    return str(isd_file)

@pytest.fixture
def mock_trade_journals(tmp_path):
    """建立臨時的 trade_journals.json"""
    journal_data = {
        "journals": {
            "#2_HUMAN_GOLD_STANDARD": [
                {
                    "wave": "Wave 1",
                    "buy_date": "2026-01-02",
                    "sell_date": "2026-01-13",
                    "buy_price": 500.0,
                    "sell_price": 600.0,
                    "shares": 100,
                    "realized_pnl": 10000.0,
                    "pnl_pct": 20.0
                }
            ],
            "A8_TOP_DEFENSE_MODERATE": [
                {
                    "agent_id": "A8_TOP_DEFENSE_MODERATE",
                    "date": "2026-01-15",
                    "action": "BUY",
                    "price": 500.0,
                    "shares": 200,
                    "amount": 100000.0,
                    "reason": "HIGH_CONFIDENCE_BYPASS"
                },
                {
                    "agent_id": "A8_TOP_DEFENSE_MODERATE",
                    "date": "2026-02-02",
                    "action": "SELL",
                    "price": 460.0,
                    "shares": 200,
                    "amount": 92000.0,
                    "reason": "HARD_STOP_LOSS_8PCT"
                }
            ]
        }
    }
    journal_file = tmp_path / "trade_journals.json"
    with open(journal_file, "w", encoding="utf-8") as f:
        json.dump(journal_data, f, indent=2, ensure_ascii=False)
    return str(journal_file)

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


def test_parse_trade_journals_success(mock_trade_journals):
    """測試解析交易日誌成功情境 (且確認去個人帳本化：不含價格與持股數)"""
    res = _parse_trade_journals(mock_trade_journals)
    assert "完成 1 個波段交易" in res["human_summary"]
    assert "完成 1 次往返交易" in res["a8_summary"]
    assert "停損" in res["a8_summary"]
    # 徹底去個人帳本化
    assert "買入價格" not in res["a8_summary"]
    assert "股數" not in res["a8_summary"]

def test_parse_trade_journals_missing_file():
    """測試當交易日誌不存在時，應能優雅保底"""
    res = _parse_trade_journals("non_existent_path.json")
    assert "歷史波段：已完成多波段" in res["human_summary"]
    assert "歷史波段：完成多波段" in res["a8_summary"]


@patch("src.ai_analyzer.genai.Client")
def test_ai_analyzer_success_buy(mock_client_class, mock_trade_journals, mock_isd_report_buy, valid_features, temp_report_file):
    """情境 1：API 成功，且今日為 BUY 訊號，進場計畫正常生成"""
    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 65,
      "action": "【🎯 進場買進】",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $692.6 (-3.0%) 考慮進場",
      "exit_plan": "達 $749.7 (+5.0%) 停利 / 跌破 $678.3 (-5.0%) 嚴格停損",
      "defense_plan_empty_hand": "不適用",
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
        assert res["action"] == "【🎯 進場買進】"
        assert res["suggested_position"] == 0.1
        assert "692.6" in res["entry_plan"]
        assert res["llm_fallback"] is False
        assert "A8_TOP_DEFENSE_MODERATE" in res["champion_agent"]


@patch("src.ai_analyzer.genai.Client")
def test_ai_analyzer_hold_suppression(mock_client_class, mock_trade_journals, mock_isd_report_hold, valid_features, temp_report_file):
    """情境 2：今日為 HOLD 訊號 (非買點)，進場計畫強制為 None 且建議倉位強制為 0.0"""
    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 50,
      "action": "【🛡️ 持股續抱】",
      "suggested_position": 0.0,
      "entry_plan": null,
      "exit_plan": "達 $749.7 (+5.0%) 停利 / 跌破 $678.3 (-5.0%) 嚴格停損",
      "defense_plan_empty_hand": "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "波段運行中持股續抱。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["action"] == "【🛡️ 持股續抱】"
        assert res["suggested_position"] == 0.0
        assert res["entry_plan"] is None
        assert "非標準買點" in res["defense_plan_empty_hand"]
        assert res["llm_fallback"] is False


@patch("src.ai_analyzer.genai.Client")
def test_ai_analyzer_missing_champion_summary(mock_client_class, mock_trade_journals, valid_features, temp_report_file, capsys):
    """情境 3：isd_predict_report.json 缺失 champion_summary，輸出 [Warning] 並保底"""
    # 建立一個不含 champion_summary 的 ISD
    isd_data = {"isdTriggered": False}
    isd_file = os.path.join(os.path.dirname(temp_report_file), "isd_predict_report.json")
    with open(isd_file, "w", encoding="utf-8") as f:
        json.dump(isd_data, f)

    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 50,
      "action": "【🛡️ 持股續抱】",
      "suggested_position": 0.0,
      "entry_plan": null,
      "exit_plan": "達 $749.7 (+5.0%) 停利 / 跌破 $678.3 (-5.0%) 嚴格停損",
      "defense_plan_empty_hand": "波段運行中，非標準買點嚴禁追高，耐性等待下一輪量化訊號",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "保底診斷。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["champion_agent"] == "A8_TOP_DEFENSE_MODERATE"
        assert res["action"] == "【🛡️ 持股續抱】"
        
        # 驗證 sys.stderr
        captured = capsys.readouterr()
        assert "champion_summary missing" in captured.err


@patch("src.ai_analyzer.genai.Client")
@patch("time.sleep")
def test_ai_analyzer_cascade_success(mock_sleep, mock_client_class, mock_trade_journals, mock_isd_report_buy, valid_features, temp_report_file, capsys):
    """情境 4：首選模型 (gemini-3.7-flash) 調用失敗，自動無縫降級切換至備援模型 (gemini-3.6-flash) 且成功"""
    mock_response_text = """
    {
      "date": "2026-08-17",
      "win_rate": 65,
      "action": "【🎯 進場買進】",
      "suggested_position": 0.1,
      "entry_plan": "回測至 692.6 (-3.0%) 考慮進場",
      "exit_plan": "達 749.7 (+5.0%) 停利 / 跌破 678.3 (-5.0%) 嚴格停損",
      "defense_plan_empty_hand": "不適用",
      "gap_defense_note": "若隔日遭遇極端跳空開盤，原設定價位立即失效，嚴禁追價",
      "wang_mou_analysis": "主力多頭洗盤，指標高檔，王謀定性建議小資金摸底試單。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("503 UNAVAILABLE"),
        MagicMock(text=mock_response_text)
    ]
    mock_client_class.return_value = mock_client
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
        res = run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-17"
        assert res["close"] == 714.0
        assert res["action"] == "【🎯 進場買進】"
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
def test_ai_analyzer_cascade_fail_throws_exception(mock_sleep, mock_client_class, mock_trade_journals, mock_isd_report_buy, valid_features, temp_report_file, capsys):
    """情境 5：主備模型皆失敗，Fail-Fast 直接拋出例外中斷且向 sys.stderr 輸出具體 [ERROR] 與 Traceback"""
    mock_client = MagicMock()
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
    """情境 6：當 GEMINI_API_KEY 缺失時，立即拋出 ValueError("GEMINI_API_KEY 未設定") 並向 sys.stderr 輸出錯誤"""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with pytest.raises(ValueError, match="GEMINI_API_KEY 未設定"):
            run_ai_analysis(latest_features=valid_features, report_path=temp_report_file)
            
        captured = sys.stderr.getvalue() if hasattr(sys.stderr, "getvalue") else capsys.readouterr().err
        assert "[ERROR] GEMINI_API_KEY 未設定" in captured
