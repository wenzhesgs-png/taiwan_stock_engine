import sys
import os
import json

# 強制將當前目錄加入 Python 搜尋路徑中
sys.path.append(os.path.dirname(__file__))

# 引入五大模組
from data_preprocessing import execute_v34_preprocessing_with_update
from train_model import execute_v34_closed_loop_rolling_train
from predict_today import predict_realtime_v_turn
from simulate_agents import run_agent_arena_simulation
from llm_agent import run_llm_agent_decision
from notify import send_telegram_notification

def run_central_tactical_pipeline():
    print("🛸" + "═"*65)
    print("🔥 [軍師控制艙啟動] 正在執行瑞昱 (2379.TW) 一鍵全自動量化與大腦管線...")
    print("═"*67 + "\n")
    
    # Step 1：特徵工程
    print("📡 Step 1/6: 正在進行聯網採集與籌碼特徵洗滌...")
    execute_v34_preprocessing_with_update()
    print("\n⚡ " + "═"*45 + "\n")
    
    # Step 2：模型重訓
    print("🧠 Step 2/6: 正在執行大腦 R2R 滾動反思特訓...")
    execute_v34_closed_loop_rolling_train()
    print("\n⚡ " + "═"*45 + "\n")

    # Step 3：當日推估/風控
    print("🔮 Step 3/6: 正在進行當日 AVM 與 ISD 智慧風控探測...")
    quant_data = predict_realtime_v_turn()
    print("\n⚡ " + "═"*45 + "\n")

    # Step 4：Agent 沙盒模擬
    print("🏆 Step 4/6: 正在啟動 9 大 Agent 沙盒競技場推演...")
    run_agent_arena_simulation()
    print("\n⚡ " + "═"*45 + "\n")

    # Step 5：LLM 定性裁決
    print("🦅 Step 5/6: 正在將冠軍基因與盲測數據移交給首席軍師 王謀...")
    
    # 讀取 Step 4 生產的模擬結果檔案，提取關鍵數據 (Context Pruning)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sim_results_path = os.path.abspath(os.path.join(current_dir, "..", "data", "simulation_results.json"))
    
    isd_triggered = False
    benchmark_roi = 0.0
    champion_agent_roi = 0.0
    today_decision = {"action": "HOLD", "position": 0.0}
    
    if os.path.exists(sim_results_path):
        try:
            with open(sim_results_path, "r", encoding="utf-8") as f:
                sim_data = json.load(f)
                isd_triggered = sim_data.get("isdTriggered", False)
                
                agents = sim_data.get("agents", [])
                # 提取 #0 Buy & Hold 基準組 ROI
                benchmark_agent = next((a for a in agents if a.get("agent_id") == "Agent_0"), None)
                if benchmark_agent:
                    benchmark_roi = float(benchmark_agent["backtest"]["roi_pct"]) / 100.0
                    
                # 排除 Agent_0，在 9 大實體 Agent 中尋找冠軍
                real_agents = [a for a in agents if a.get("agent_id") != "Agent_0"]
                if real_agents:
                    champion_agent = max(real_agents, key=lambda x: float(x["backtest"]["roi_pct"]))
                    champion_agent_roi = float(champion_agent["backtest"]["roi_pct"]) / 100.0
                    today_decision = {
                        "action": champion_agent["today_decision"]["action"],
                        "position": champion_agent["today_decision"]["position"]
                    }
                    print(f"🔥 [冠軍 Agent 提取成功]：{champion_agent['name']} 奪冠，回測 ROI = +{champion_agent['backtest']['roi_pct']}%！")
        except Exception as e:
            print(f"⚠️ 提取競技場冠軍基因時出錯 ({e})，使用保底量化數據進行 LLM 診斷。")
    else:
        print("⚠️ 找不到 simulation_results.json，使用預設空值數據。")

    # 呼叫 llm_agent 裁決
    report_data = run_llm_agent_decision(
        isd_triggered=isd_triggered,
        benchmark_roi=benchmark_roi,
        champion_agent_roi=champion_agent_roi,
        today_decision=today_decision
    )
    
    # 【一鍵自動化測試與閉環保障防線】
    # 當在 TDD 測試下 (Step 5 被 Mock 時)，親自、顯式確保 data/llm_agent_report.json 被寫入
    try:
        report_path = os.path.abspath(os.path.join(current_dir, "..", "data", "llm_agent_report.json"))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ main.py 顯式備份寫入 llm_agent_report.json 失敗 ({e})")
    
    print("\n" + "═"*70)
    print("🦅 【首席軍師 王謀 ｜ 終極定性戰術報告 JSON 導出】")
    print("═"*70)
    print(json.dumps(report_data, indent=2, ensure_ascii=False))
    print("═"*70 + "\n")
    
    # Step 6：Telegram 戰術報告推播
    print("📢 Step 6/6: 正在執行 Telegram 戰術報告推播與閉環通知...")
    pushed = send_telegram_notification(report_path=report_path)
    if pushed:
        print("✅ [推播狀態] 成功送出 Telegram 頻道通知。")
    else:
        print("⚠️ [推播狀態] 推播已跳過或發送失敗 (無憑證或連線異常)。")
    print("\n⚡ " + "═"*45 + "\n")
    
    print("═"*67)
    print("🏁 [任務全線通關] 數據更新、大腦迭代、競技場模擬、王謀裁決與 Telegram 通知一鍵閉環全自動完成！")
    print("═"*67 + "\n")

if __name__ == "__main__":
    run_central_tactical_pipeline()
