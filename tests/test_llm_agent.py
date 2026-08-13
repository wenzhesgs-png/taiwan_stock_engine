import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# 將 project root 加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.llm_agent
from src.llm_agent import run_llm_agent_decision

def test_llm_agent_decision_success():
    """
    測試 LLM 裁決引擎在正常 API 回覆下的決策生成
    """
    mock_response_text = """
    {
      "action": "進場",
      "suggested_position": 1.0,
      "wang_mou_analysis": "主力假摔洗盤，籌碼暗中吸籌完畢，建議強勢買進試單。"
    }
    """
    
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=mock_response_text)
    
    # 物理篡改全域 client，100% 免疫 API 連線與 Mock 順序問題
    src.llm_agent.client = mock_client
    # 保障其 API KEY 為有效狀態
    src.llm_agent.API_KEY = "valid_key"
         
    res = run_llm_agent_decision(
        isd_triggered=False,
        benchmark_roi=0.4053,
        champion_agent_roi=1.2482,
        today_decision={"action": "BUY", "position": 1.0}
    )
    
    assert res["action"] == "進場"
    assert res["suggested_position"] == 1.0
    assert "wang_mou_analysis" in res
    assert res["llm_fallback"] is False


def test_llm_agent_isd_override():
    """
    測試當 isd_triggered == True 時，Python 程式碼層必須一票否決，
    強制將 action 覆蓋為 '觀望'，suggested_position 設為 0.0，且 wang_mou_analysis 明確標記風控。
    （不應發送任何 API 呼叫）
    """
    mock_client = MagicMock()
    src.llm_agent.client = mock_client
    
    res = run_llm_agent_decision(
        isd_triggered=True,
        benchmark_roi=0.4053,
        champion_agent_roi=1.2482,
        today_decision={"action": "BUY", "position": 1.0}
    )
    
    assert res["action"] == "觀望"
    assert res["suggested_position"] == 0.0
    assert "ISD" in res["wang_mou_analysis"] or "熔斷" in res["wang_mou_analysis"]
    assert res["llm_fallback"] is False
    # 驗證未調用外部 API
    mock_client.models.generate_content.assert_not_called()


def test_llm_agent_timeout_fallback():
    """
    測試當 API 呼叫超時（超過 10s）或拋出錯誤時，Fail-Safe 靜態降級機制。
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Timeout after 10s")
    src.llm_agent.client = mock_client
    src.llm_agent.API_KEY = "valid_key"
     
    res = run_llm_agent_decision(
        isd_triggered=False,
        benchmark_roi=0.4053,
        champion_agent_roi=1.2482,
        today_decision={"action": "BUY", "position": 1.0}
    )
    
    assert res["action"] == "觀望"
    assert res["suggested_position"] == 0.0
    assert "降級" in res["wang_mou_analysis"] or "維持觀望" in res["wang_mou_analysis"]
    assert res["llm_fallback"] is True


def test_llm_agent_missing_key_fallback():
    """
    測試當 GEMINI_API_KEY 缺失時，自動觸發降級，不崩潰。
    """
    src.llm_agent.API_KEY = ""
    res = run_llm_agent_decision(
        isd_triggered=False,
        benchmark_roi=0.4053,
        champion_agent_roi=1.2482,
        today_decision={"action": "BUY", "position": 1.0}
    )
    assert res["action"] == "觀望"
    assert res["suggested_position"] == 0.0
    assert res["llm_fallback"] is True
