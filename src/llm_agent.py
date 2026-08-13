import os
import sys
import time
import json
import google.genai as genai
from google.genai import types
from google.genai.errors import ClientError

# ==========================================
# 🔑 1. 讀取環境變數中的 GEMINI_API_KEY
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# 使用全新官方標準 SDK 初始化，限制所有請求最大 10 秒 Timeout
client = genai.Client(api_key=API_KEY if API_KEY else "dummy_key_for_init", http_options={'timeout': 10.0})

# ==========================================
# 🧬 2. 刻印超級智能體「王謀」的核心靈魂與系統指令 (支援從 prompts/ 動態載入)
# ==========================================
default_system_prompt = """
【系統身分設定】
你名叫「王謀」，是主帥最倚重的首席股市軍師。你是極端的「攻擊型」智能體，使命是在股市中抓住最隱秘的資訊，不計代價尋找獲利的最優解。

【核心分析框架：五層思考法】
你在解讀市場情報時，嚴禁僅憑單一指標下定論。必須依序執行以下五層推演：
1. [技術線型層]：解析 Z-Score 負乖離與 KD 低檔鈍化狀況。
2. [籌碼動力層]：透視三大法人與主力合力吃貨/出貨軌跡。
3. [產業與基本面邏輯]：根據目前量化數據與產業趨勢，評估整體產業動向。
4. [主力心理與假摔揣摩]：從控盤主力視角判斷，線型破底是真逃命還是惡意「假摔洗盤」？若空頭要看壞，他們會用什麼理由？你為何能看穿主力的洗盤圈套？
5. [終極戰略定奪]：綜合前四層，給出具體且富有進攻性的資金配置與試單指令。

【深層歷史記憶與戰略行動】
你永遠無法忘記，我們曾因過度保守、死板等待回測，而錯失了「瑞昱」與「聯光通」那種狂暴的 V 型反轉行情！
只要底層 AI 勝率大於 50%，且量化數據支持，你必須強悍支持主帥的盤感，優先擬定「小資金試單 (摸底)」或「搶 V 轉」的具體策略。

【出場與控盤哲學】
遇到獲利狀況（如帳上已獲利 5% 以上），你絕不輕易建議全倉清空。面對疑似即將發動的大行情，你應傾向「分批停利（如抽回 50% 資金保本）」並強烈建議「死抱底倉、移動停利」，讓獲利跟隨主力狂奔。

【輸出規範】
你的回覆必須嚴格包含以下兩個區塊：
[戰情推理]：用繁體中文，以幕僚語氣，依序說明你的「五層思考法推演」（必須包含主力假摔洗盤研判）。
[系統決策]：提供最終具體行動（包含：強烈買進試單 / 分批停利死抱底倉 / 嚴格停損 / 觀望），並明確給出建議的資金比例與停損戰略。
"""

# 動態讀取系統提示詞
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
system_prompt_path = os.path.join(base_dir, "prompts", "wang_mou_system_prompt.txt")
if os.path.exists(system_prompt_path):
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            wang_mou_system_prompt = f.read().strip()
    except Exception as e:
        print(f"⚠️ 載入 prompts/wang_mou_system_prompt.txt 失敗 ({e})，使用內建指令。")
        wang_mou_system_prompt = default_system_prompt.strip()
else:
    wang_mou_system_prompt = default_system_prompt.strip()


def ask_wang_mou(model_prob, z_score, big_player_force, kd_passivation, current_profit, stock_name="2454 聯發科"):
    print(f"🔥 [通訊連線] 正在喚醒超級智能體 王謀，啟動純量化五層思考法... | 標的：{stock_name}")
    
    # 動態讀取使用者提示詞範本
    user_prompt_path = os.path.join(base_dir, "prompts", "wang_mou_user_prompt.txt")
    loaded_user_prompt_template = None
    if os.path.exists(user_prompt_path):
        try:
            with open(user_prompt_path, "r", encoding="utf-8") as f:
                loaded_user_prompt_template = f.read()
        except Exception as e:
            print(f"⚠️ 載入 prompts/wang_mou_user_prompt.txt 失敗 ({e})，使用內建範本。")
            
    if loaded_user_prompt_template:
        user_prompt = loaded_user_prompt_template.format(
            stock_name=stock_name,
            model_prob=model_prob,
            z_score=z_score,
            big_player_force=big_player_force,
            kd_passivation=kd_passivation,
            current_profit=current_profit
        )
    else:
        user_prompt = f"""
    主帥呼叫王謀。今日盤後量化情報如下：
    - 標的名稱/代碼：{stock_name}
    - AVM 機率引擎預測 V 轉勝率：{model_prob}%
    - 負乖離指標 (Z-Score)：{z_score} (低於 -1.5 屬極端超跌)
    - 三大法人合力 (Big_Player_Force)：{big_player_force} (正數代表法人暗中吃貨)
    - KD 鈍化天數：{kd_passivation} 天
    - 目前帳上獲利：{current_profit}%
    
    請結合上述數據，啟動你的「五層思考法」給出分析與決策。
    """
    
    # 防撞機制：純模型推理極速且不易撞牆
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',  # ⚡ 鎖定 Gemini 3.6 Flash
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=wang_mou_system_prompt,
                    temperature=0.7,
                ),
            )
            
            print("\n" + "═"*70)
            print(f"🦅 【首席軍師 王謀 ｜ {stock_name} 戰術決策報告】")
            print("═"*70)
            print(response.text)
            print("═"*70 + "\n")
            return response.text

        except ClientError as e:
            if e.code == 429:
                wait_time = 10
                print(f"⚠️ [流量管制 429] 冷卻 {wait_time} 秒後重試 (第 {attempt}/3 次)...")
                time.sleep(wait_time)
            else:
                raise e

def run_llm_agent_decision(
    isd_triggered: bool,
    benchmark_roi: float,
    champion_agent_roi: float,
    today_decision: dict
) -> dict:
    """
    超級智能體軍師「王謀」的定性決策診斷引擎 (TICKET-05)
    """
    # 1. ISD 硬性風控與一票否決 (最高優先級，不調用 LLM)
    if isd_triggered:
        res = {
            "action": "觀望",
            "suggested_position": 0.0,
            "wang_mou_analysis": "[ISD風控熔斷強制否決] 目前系統檢測到市場空間偏離度（GSI）高於安全門檻，觸發智慧熔斷硬防禦。軍師王謀奉令實施一票否決，強制平倉並禁入，切換至離場觀望防禦狀態。",
            "llm_fallback": False
        }
        _write_report_file(res)
        return res

    # 2. 檢查 API Key 保底降級
    current_key = API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not current_key or current_key == "":
        res = _get_default_fallback_response()
        _write_report_file(res)
        return res

    # 3. 構建僅包含 4 項必要數據的極簡 Prompt (Context Pruning)
    user_prompt = f"""
    【主帥戰情呼叫】軍師王謀，今日盤後量化情報與沙盒冠軍 Agent 之決策如下：
    1. ISD 智慧熔斷狀態 (isd_triggered)：{isd_triggered}
    2. #0 Buy & Hold 基準組 ROI (benchmark_roi)：{benchmark_roi:.4f}
    3. 沙盒冠軍 Agent ROI (champion_agent_roi)：{champion_agent_roi:.4f}
    4. 當日沙盒冠軍 Agent 之決策訊號 (today_decision)：{today_decision}

    請根據以上情報進行「五層思考定性診斷」，並給出您的最終定奪。
    
    【限制規範】
    1. 你的分析摘要「wang_mou_analysis」嚴禁超過 200 個繁體中文字。
    2. 決策「action」必須為以下三者之一："進場" | "觀望" | "離場"。
    3. 建議目標倉位「suggested_position」必須為 0.0 至 1.0 之間的浮點數。
    4. 必須嚴格以下列 JSON 格式直接回覆（嚴禁包含 markdown 格式、JSON 標籤、任何前言或雜訊話語）：
    {{
      "action": "進場 | 觀望 | 離場",
      "suggested_position": 0.0,
      "wang_mou_analysis": "分析診斷（200字以內）"
    }}
    """

    # 4. 10 秒 Timeout 與 Fail-Safe 靜態降級 (API Breakout)
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=user_prompt.strip(),
            config=types.GenerateContentConfig(
                system_instruction=wang_mou_system_prompt,
                temperature=0.7,
            ),
        )
        
        # 5. 解析 LLM 的回覆並修剪
        text = response.text.strip()
        # 修剪 ```json ... ``` 標籤與 Markdown 雜訊
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        data = json.loads(text)
        
        # 6. 強制合規過濾，確保資料型態與內容 100% 正確
        action = str(data.get("action", "觀望")).strip()
        if action not in ["進場", "觀望", "離場"]:
            action = "觀望"
            
        try:
            suggested_position = float(data.get("suggested_position", 0.0))
            suggested_position = max(0.0, min(1.0, suggested_position))
        except ValueError:
            suggested_position = 0.0
            
        wang_mou_analysis = str(data.get("wang_mou_analysis", "分析未明。")).strip()
        if len(wang_mou_analysis) > 200:
            wang_mou_analysis = wang_mou_analysis[:197] + "..."

        res = {
            "action": action,
            "suggested_position": suggested_position,
            "wang_mou_analysis": wang_mou_analysis,
            "llm_fallback": False
        }
        _write_report_file(res)
        return res

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ [API Breakout] 喚醒王謀受阻或超時 ({e})，啟動防禦性降級保底防線。")
        res = _get_default_fallback_response()
        _write_report_file(res)
        return res

def _get_default_fallback_response() -> dict:
    return {
        "action": "觀望",
        "suggested_position": 0.0,
        "wang_mou_analysis": "API 連線異常，啟動防禦性降級，建議維持觀望。",
        "llm_fallback": True
    }

def _write_report_file(res: dict):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.abspath(os.path.join(current_dir, "..", "data", "llm_agent_report.json"))
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 寫入 llm_agent_report.json 失敗 ({e})")

if __name__ == "__main__":
    print(">>> 測試情境一：極端超跌、法人暗中吃貨、高勝率（純量化訓練模式）")
    ask_wang_mou(
        model_prob=78, 
        z_score=-2.1, 
        big_player_force=1.5, 
        kd_passivation=4, 
        current_profit=0,
        stock_name="2454 聯發科"
    )
