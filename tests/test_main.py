import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# 將 project root 加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_central_pipeline_closed_loop_execution():
    """
    測試主管線一鍵全自動量化大腦與 LLM 控制調度艙 (main.py) 的閉環執行
    """
    from src.main import run_central_tactical_pipeline
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.abspath(os.path.join(script_dir, "..", "data", "llm_agent_report.json"))
    
    # 確保原本檔案被清除，以防干擾測試
    if os.path.exists(report_path):
        os.remove(report_path)
        
    mock_llm_decision = {
        "action": "進場",
        "suggested_position": 1.0,
        "wang_mou_analysis": "主力在多頭均線上方強勢洗盤，籌碼面主力法人大買，戰術定奪：All-In滿倉，跟隨獲利奔跑。",
        "llm_fallback": False
    }
    
    # 模擬 5 大階段，將 Step 1~4 的模組進行 Mock
    with patch("src.main.execute_v34_preprocessing_with_update") as mock_step1, \
         patch("src.main.execute_v34_closed_loop_rolling_train") as mock_step2, \
         patch("src.main.run_agent_arena_simulation") as mock_step3, \
         patch("src.main.predict_realtime_v_turn") as mock_step4, \
         patch("src.main.run_llm_agent_decision") as mock_step5:
         
         # 設置模擬 Step 4 返回數據
         mock_step4.return_value = {
             "stock_name": "2379 瑞昱",
             "model_prob": 0.85,
             "z_score": 1.2,
             "big_player_force": 2.5,
             "kd_passivation": 3,
             "current_profit": 0.0
         }
         
         # 設置 Step 5 定性裁決 Mock 返回
         mock_step5.return_value = mock_llm_decision
         
         # 執行主管線一鍵調度
         run_central_tactical_pipeline()
         
         # 斷言五大模組都順利被呼叫，完成自動化一鍵式閉環！
         mock_step1.assert_called_once()
         mock_step2.assert_called_once()
         mock_step3.assert_called_once()
         mock_step4.assert_called_once()
         mock_step5.assert_called_once()
         
         # 驗證 main.py 調度完後，是否有將最終裁決寫入 data/llm_agent_report.json 檔案中
         assert os.path.exists(report_path), "llm_agent_report.json 應被成功創建"
         
         with open(report_path, "r", encoding="utf-8") as f:
             report_data = json.load(f)
             
         assert report_data["action"] == "進場"
         assert report_data["suggested_position"] == 1.0
         assert "wang_mou_analysis" in report_data
         assert report_data["llm_fallback"] is False
