import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.ai_analyzer import run_ai_analysis, compute_technical_summary, _parse_raw_price_data

@pytest.fixture
def mock_raw_data(tmp_path):
    """建立臨時的 raw_2379.csv 數據"""
    raw_file = tmp_path / "raw_2379.csv"
    raw_content = (
        "Price,Close,High,Low,Open,Volume\n"
        "Ticker,2379.TW,2379.TW,2379.TW,2379.TW,2379.TW\n"
        "Date,,,,,\n"
        "2026-08-11,772.0,780.0,758.0,758.0,2282530\n"
        "2026-08-12,766.0,776.0,760.0,772.0,1486403\n"
    )
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(raw_content)
    return str(raw_file)

@pytest.fixture
def temp_report_file(tmp_path):
    return str(tmp_path / "llm_agent_report.json")

def test_parse_raw_price_data(mock_raw_data):
    """測試解析 yfinance raw 數據的正確性"""
    df = _parse_raw_price_data(mock_raw_data)
    assert not df.empty
    assert len(df) == 2
    assert "Close" in df.columns
    assert df.iloc[-1]["Close"] == 766.0

def test_compute_technical_summary(mock_raw_data):
    """測試 KD/MACD 狀態摘要計算"""
    df = _parse_raw_price_data(mock_raw_data)
    close_val, tech_summary = compute_technical_summary(df)
    assert close_val == 766.0
    assert "KD" in tech_summary
    assert "MACD" in tech_summary

@patch("google.genai.Client")
def test_ai_analyzer_success(mock_genai_client_class, mock_raw_data, temp_report_file):
    """情境 1：API 調用成功，返回完整的符合 Schema 的 JSON 數據"""
    # 模擬 API 回覆
    mock_response_text = """
    {
      "date": "2026-08-12",
      "win_rate": 65,
      "action": "小資金試單 (搶V轉)",
      "suggested_position": 0.1,
      "entry_plan": "回測至 $750 (-2.1%) 考慮進場 / 跌破 $740 (-3.4%) 必買",
      "exit_plan": "達 $800 (+4.4%) 停利 / 跌破 $730 (-4.7%) 嚴格停損",
      "gap_defense_note": "若開盤跳空開低於停損價或大幅跳空開高，本計畫失效，嚴禁盲目追價",
      "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，KD指標低檔超賣，建議執行防禦型摸底進場。"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    
    with patch("src.ai_analyzer.client", mock_client), \
         patch("src.ai_analyzer.API_KEY", "valid_key"):
        
        res = run_ai_analysis(raw_data_path=mock_raw_data, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-12"
        assert res["win_rate"] == 65
        assert res["action"] == "小資金試單 (搶V轉)"
        assert res["suggested_position"] == 0.1
        assert "750" in res["entry_plan"]
        assert "800" in res["exit_plan"]
        assert "跳空" in res["gap_defense_note"]
        assert "主力在多頭均線" in res["wang_mou_analysis"]
        assert res["llm_fallback"] is False
        assert res["close"] == 766.0
        assert "KD" in res["technical_summary"]

@patch("google.genai.Client")
def test_ai_analyzer_schema_missing_fields_remedy(mock_genai_client_class, mock_raw_data, temp_report_file):
    """情境 2：API 回傳 JSON 缺失部分欄位，測試預設值補齊與保護"""
    # 缺失部分欄位，如 win_rate, suggested_position, gap_defense_note 等
    mock_response_text = """
    {
      "date": "2026-08-12",
      "action": "小資金試單",
      "entry_plan": "回測至 $750 進場",
      "exit_plan": "達 $800 停利",
      "wang_mou_analysis": "分析內容"
    }
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    
    with patch("src.ai_analyzer.client", mock_client), \
         patch("src.ai_analyzer.API_KEY", "valid_key"):
        
        res = run_ai_analysis(raw_data_path=mock_raw_data, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-12"
        assert res["win_rate"] == 50  # 補齊之預設值
        assert res["action"] == "小資金試單"
        assert res["suggested_position"] == 0.0  # 補齊之預設值
        assert res["entry_plan"] == "回測至 $750 進場"
        assert "跳空" in res["gap_defense_note"]  # 補齊之防守警語
        assert res["wang_mou_analysis"] == "分析內容"
        assert res["llm_fallback"] is False

@patch("google.genai.Client")
def test_ai_analyzer_api_error_fallback(mock_genai_client_class, mock_raw_data, temp_report_file):
    """情境 3：API 呼叫失敗或超時，觸發防禦性降級 (llm_fallback: True)"""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Timeout Error")
    
    with patch("src.ai_analyzer.client", mock_client), \
         patch("src.ai_analyzer.API_KEY", "valid_key"):
        
        res = run_ai_analysis(raw_data_path=mock_raw_data, report_path=temp_report_file)
        
        assert res["date"] == "2026-08-12"
        assert res["win_rate"] == 50
        assert res["action"] == "觀望"
        assert res["suggested_position"] == 0.0
        assert "考慮進場" in res["entry_plan"]
        assert "嚴格停損" in res["exit_plan"]
        assert "跳空" in res["gap_defense_note"]
        assert "防禦性降級" in res["wang_mou_analysis"]
        assert res["llm_fallback"] is True
