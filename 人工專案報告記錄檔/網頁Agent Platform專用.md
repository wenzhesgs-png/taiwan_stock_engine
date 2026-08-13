# 專案架構現況、記憶檔與完整任務單 (Project Memory & Full Ticket Specs)

> **專案核心現況註記（重要）**：
> 本專案底層 AVM 核心架構（特徵管線、PCA/ART2 品質閘門 $DQI_x/DQI_y$、雙大腦 $M_{ref}/M_{conj}$、風控熔斷 $RI/GSI/ISD$）**皆已完成實作與測試**。
> 目前專案阻礙已精確鎖定於 **「TICKET-04 策略層磨損與過度濾波 (Strategy Attrition)」** 與 **「TICKET-05 LLM 調度閉環」**。

---

## 📊 模組驗收狀態一覽表 (Module Status Overview)

| 模組檔案 | 版本 | 驗收狀態 | 說明 / 現況 |
| :--- | :--- | :--- | :--- |
| `src/data_preprocessing.py` | V3.4 | **✅ 100% 驗收完成** | 12 維技術特徵 + 5 維籌碼特徵 + $Y_{Alpha}$ 標籤；硬性套用 `.shift(1)` 徹底封死前瞻偏誤 (Look-ahead Bias)。 |
| `src/train_model.py` | V3.4 | **✅ 100% 驗收完成** | 雙模型 ($M_{ref}$ 3年 + $M_{conj}$ 滾動) + $GSI$ Ridge 正則化；已完成頂層 `sys.path` 路徑修護（已無 Warning）。 |
| `src/predict_today.py` | V3.4 | **✅ 100% 驗收完成** | 雙大腦推估 + $DQI_x$ 歐氏距離 + $RI$ 重疊面積 + $GSI$ 馬氏距離，完整實作 ISD 智慧熔斷開關。 |
| `src/simulate_agents.py` | V5.3 | **🔥 進行中 (Active)** | 9 大 Agent 量化沙盒。實施「風險分級高勝率旁路」與「三軌離場機制」，消滅微幅高頻交易損耗。 |
| `src/llm_agent.py` | V3.6 | **🔶 待辦 (Pending)** | 軍師王謀「五層思考法」定性診斷 + 10 秒 Timeout 靜態降級斷路器 + Context Pruning。 |
| `src/main.py` | V3.6 | **🔶 待辦 (Pending)** | 全主管線一鍵自動化調度艙，貫穿 Steps 1~5 輸出最終閉環報告。 |

---

# 系統技術規格書 (System Specification)

## 1. Goal & Context (目標與背景)
本專案旨在將半導體/面板高階製造業之「全自動虛擬量測 (AVM)」技術跨界融合至台股 AI 股市預測與決策系統。消滅盤中與盤後資訊延遲痛點，建立一個實時 (Real-time)、具備資料品質雙重驗證 ($DQI_x/DQI_y$)、雙大腦預測 ($M_{ref}/M_{conj}$)、風險熔斷 ($RI/GSI/ISD$) 與雙階段 (Dual-Phase) 線上進化的自動化股票虛擬量測決策引擎。

當前版本重點為透過 V5.3 重構修復 `simulate_agents.py` 之策略層磨損問題，將大腦高勝率預測精確轉化為 **ROI $\ge +120\%$** 且交易次數 **$\le 10$ 次** 的實體獲利，並串接 `llm_agent.py` 與 `main.py` 達成一鍵閉環。

## 2. Non-Goals (非目標)
* **極高頻交易 (HFT) 下單執行**：專注於虛擬量測推估與訊號輸出，不涵蓋微秒級券商 API 報價搶單。
* **前端視覺化 UI 介面**：僅涵蓋核心後端服務、資料管線、模型運算引擎與 API 介面。
* **無節制高頻交易**：不追求交易頻率，嚴格控制總交易次數 $\le 10$ 次以壓制交易成本。

## 3. Architecture Seams & Data Flow (系統架構與狀態)

[ Tick/Daily Data ] --> [ Data Pipeline & Anti-Leakage Shift ] (✅ COMPLETED V3.4)
|
v
[ Feature Extractor (17D Features) ] (✅ COMPLETED V3.4)
|
v
[ DQI_x Quality Gate (PCA + LOO) ] --(Fail)--> [ Reject / Log ]
| (Pass)
v
[ Stock Group Router & Dual Model ] (✅ COMPLETED V3.4)
(M_ref & M_conj)
|
v
[ Risk Engine (RI & GSI) ] --(Unsafe)--> [ ISD Circuit Breaker ] (✅ COMPLETED V3.4)
| (Safe)
v
[ Phase I Intraday Prediction API / ISD Report ] (✅ COMPLETED V3.4)
|
v
[ V5.3 Strategy Engine & Sandbox Simulation ] (🔥 ACTIVE NEXT STEP)
(High-Confidence Bypass & Friction Control)
|
v
[ LLM Strategist & Master Pipeline Controller ] (🔶 PENDING V3.6)
|
(Post-market Ground Truth)
v
[ DQI_y Quality Gate (ART2) ] --(Fail)--> [ Discard Label ]
| (Pass)
v
[ Phase II Dynamic Retraining Pipeline ] (✅ COMPLETED V3.4)


## 4. Acceptance Criteria (驗收標準)
1. **資料管線與防洩漏**：$T$ 日特徵矩陣之籌碼欄位 100% 取自 $T-1$ 日盤後數據（透過 `.shift(1)` 隔離），全表無 `NaN`。
2. **風控熔斷與 ISD 機制**：$RI < RI_T$ 或 $GSI > GSI_T$ 時，系統於 $10\text{ms}$ 內發出 ISD 警告，且下單建議倉位強制歸零。
3. **V5.3 策略層解鎖**：
   * 冠軍 Agent 回測 ROI **$\ge +120\%$**。
   * 歷史總交易次數 **$\le 10$ 次**，平均持股天數延伸至 **$> 10$ 天**。
4. **管線全閉環**：`python src/main.py` 能一鍵閉環貫穿，順暢輸出最終 `data/llm_agent_report.json`。

---

# 實作任務單彙整 (/to-tickets)

[ T1: Feature Pipeline & Anti-Leakage Engine ] ───────> ✅ [COMPLETED]
[ T2: Quality Gates (DQIx/DQIy) & Dual Model ] ───────> ✅ [COMPLETED]
[ T3: Risk Engine (RI/GSI) & ISD Gate API ] ──────────> ✅ [COMPLETED]
[ T-FIX: Fix train_model.py Path Noise ] ──────────────> ✅ [COMPLETED]
[ T4: V5.3 Strategy Engine & Risk-Profiled Exit ] ────> 🔥 [ACTIVE NEXT STEP]
[ T5: LLM Strategist & Master Pipeline Controller ] ──> 🔶 [PENDING]


---

### [TICKET-FIX] 消除 `src/train_model.py` 路徑載入 Warning (Fix Import Path Noise)
* **Status**: **✅ COMPLETED**
* **Strictly Affected Files**: `src/train_model.py`
* **Implementation Standard**:
  在 `src/train_model.py` 頂層注入：
  ```python
  import sys
  from pathlib import Path
  project_root = str(Path(__file__).resolve().parents[1])
  if project_root not in sys.path:
      sys.path.insert(0, project_root)
Verification: 執行 python src/main.py 時警示完全消失，pytest tests/ 全量綠燈。
[TICKET-04] V5.3 策略層與風險分級三軌離場機制重構 (Refactor V5.3 Strategy Engine & Risk-Profiled Three-Track Exit)
🎯 Objective
使命：重構 src/simulate_agents.py 內 9 大 Agent 交易策略，整合風險分級高勝率旁路（0.70 / 0.75 / 0.80）與 benchmark_exit_strategies.py 之「三軌離場機制」（軌道 3 門檻下調至 $p_{\text{conj}} < 0.40$ 防洗盤），消滅策略層過度濾波與微幅摩擦損耗，達成沙盒冠軍 Agent ROI $\ge +120%$ 且總交易次數 $\le 10$ 次。
📁 Strictly Affected Files
src/simulate_agents.py
tests/test_simulate_agents_v53.py
tests/test_simulate_agents.py (⚠️ 僅限針對與 V5.3 衝突之舊測試標記 @pytest.mark.skip)
(⚠️ Cline 警告：嚴禁讀取、搜尋或修改此清單以外的任何檔案，特別是 src/data_preprocessing.py 與 src/train_model.py)
🔗 Blocking Dependencies
TICKET-01, TICKET-02, TICKET-03 (✅ 已完成)
📐 Data Interface / Schema Definition
Input Parameters:
p_conj: float - 模型預測之多頭勝率 ($0.00 \sim 1.00$)
z_bias: float - 股價相對於 20 MA 之標準化偏離度 $Z\text{-Score}$
isd_triggered: bool - ISD 風控熔斷狀態 (True 表示觸發風控/異常)
current_position: float - 當前持有倉位比例 ($0.00 \sim 1.00$)
unrealized_pnl_pct: float - 未實現損益百分比 (例如 -0.08 代表 -8%)
agent_risk_profile: str - 戰術風險偏好 ("AGGRESSIVE" | "MODERATE" | "CONSERVATIVE")
Output Decision:
action: str - 嚴格限定為 "BUY" | "HOLD" | "EXIT"
suggested_position: float - 建議目標倉位 ($0.00 \sim 1.00$)
✅ Scope & Acceptance Criteria (DoD)
進場與風控優先級 (Entry & Risk Override)：

Given 系統給出高勝率預測且無 ISD 風控熔斷 (isd_triggered == False)
When Agent 評估進場邏輯時，依據風險偏好對應門檻：
Aggressive (A1 / A4 / A7)：$p_{\text{conj}} \ge 0.70$
Moderate (A2 / A5 / A8)：$p_{\text{conj}} \ge 0.75$
Conservative (A3 / A6 / A9)：$p_{\text{conj}} \ge 0.80$
Then 繞過一切型態過濾器（High-Confidence Bypass），輸出 action = "BUY", suggested_position = 1.0。
Given 系統觸發 ISD 智慧熔斷 (isd_triggered == True)
When Agent 評估任何交易邏輯時（無視 $p_{\text{conj}}$ 高低）
Then ISD 實施最高優先級一票否決，強制輸出 action = "EXIT", suggested_position = 0.0（當天強制平倉或禁止進場）。
Given Agent 未持倉 (current_position == 0.0) 且勝率未達該風險組別之旁路門檻
When Agent 評估進場邏輯時
Then 不觸發買進，輸出 action = "HOLD", suggested_position = 0.0。
三軌離場與持倉狀態 (Three-Track Exit Rules & Holding Zone)：

軌道 1（物理硬停損）：
Given Agent 已持倉 (current_position > 0.0) 且浮虧達到或超過 8% (unrealized_pnl_pct <= -0.08)
When 評估離場邏輯時
Then 無條件觸發，無視機率強制輸出 action = "EXIT", suggested_position = 0.0。
軌道 2（高檔過熱雙重確認）：
Given Agent 已持倉且處於高檔過熱狀態 ($z_bias \ge \text{overheat_z_bias}$，預設 2.5)
When 勝率保持強勢 ($p_{\text{conj}} \ge 0.60$) 時，輸出 action = "HOLD", suggested_position = current_position（強勢股續抱）；當且僅當勝率轉弱 ($p_{\text{conj}} < 0.60$) 時
Then 雙重確認成立，輸出 action = "EXIT", suggested_position = 0.0。
軌道 3（趨勢轉弱離場 - 優化防洗盤）：
Given Agent 已持倉，未觸發硬停損與過熱門檻 ($z_bias < \text{overheat_z_bias}$)
When 勝率持續低迷跌破轉弱門檻 ($p_{\text{conj}} < 0.40$)
Then 觸發轉弱平倉，輸出 action = "EXIT", suggested_position = 0.0。
常態續抱區間 (Normal Holding Zone)：
Given Agent 已持倉，未觸發硬停損與過熱門檻，且勝率維持安全水準 ($p_{\text{conj}} \ge 0.40$)
When 評估持倉邏輯時
Then 輸出 action = "HOLD", suggested_position = current_position。
驗收標準 (Verification Criterion)：

Given 執行全套件測試 pytest tests/
When 執行 test_simulate_agents_v53.py 與沙盒回測時
Then 單元測試 100% 通過，且沙盒冠軍 Agent ROI 精確突破 $+120%$，總交易次數自然控制於低頻極致（$\le 10$ 次）。
🛡️ Edge Cases & Fallback
[ISD 熔斷最高優先權]：ISD 風控擁有絕對最高優先權。即便 $p_{\text{conj}} = 0.99$，只要 isd_triggered == True，建議倉位必須強制歸零 0.0。
[歷史測試檔案衝突]：若 tests/test_simulate_agents.py 或歷史測試（V3.7~V5.2）因 V5.3 策略變更而斷言失敗，必須明確加上 @pytest.mark.skip(reason="Superceded by V5.3 strategy")，嚴禁修改舊版測試邏輯。
❌ Explicit Non-Goals
❌ 嚴禁修改核心檔案：嚴禁改動 src/data_preprocessing.py、src/train_model.py 與 src/predict_today.py。
❌ 嚴禁硬編碼交易次數：不得在程式碼中寫死 if trade_count > 10: break，交易頻率降低必須由「遲滯緩衝帶 + 雙重確認」機制自然達成。
❌ 嚴禁引進新套件：不得引入新的第三方依賴模組或過度設計複雜的物件繼承樹。
🛑 Execution Rules For Cline
最少改動：僅修改達成 DoD 所需的最少程式碼，嚴禁順手重構、拆分模組或調整無關樣式。
爭議即停 (Halt Immediately)：若執行過程需要修改清單外的檔案、發現欄位缺失或遇到未定義的邏輯衝突，必須聯絡工程主管停止動作並擲出報錯，嚴禁自行猜測與通靈實作。
[TICKET-05] LLM 軍師王謀定性裁決引擎與主管線一鍵自動化調度艙 (LLM Strategist & Master Controller)
🎯 Objective
使命：實作 src/llm_agent.py（整合 Gemini API、10 秒 Timeout 靜態降級斷路器、軍師王謀「五層思考法」200 字摘要）與 src/main.py（串接 Steps 1~5 主管線一鍵自動化閉環），產出最終確定性報告 data/llm_agent_report.json。
📁 Strictly Affected Files
src/llm_agent.py
src/main.py
tests/test_llm_agent.py
tests/test_main.py
(⚠️ Cline 警告：嚴禁讀取、搜尋或修改此清單以外的任何檔案，特別是 src/data_preprocessing.py、src/train_model.py、src/predict_today.py 與 src/simulate_agents.py)
🔗 Blocking Dependencies
TICKET-01, TICKET-02, TICKET-03, TICKET-04 (🔥 即將完成)
📐 Data Interface / Schema Definition
Input Parameters (llm_agent.py):
isd_triggered: bool - ISD 智慧熔斷狀態
benchmark_roi: float - #0 Buy & Hold 基準組 ROI (例如 0.4053)
champion_agent_roi: float - #1 冠軍 Agent ROI (例如 1.2482)
today_decision: dict - 當日沙盒冠軍 Agent 之決策訊號
Output Schema (data/llm_agent_report.json):
{
  "action": "進場 | 觀望 | 離場",
  "suggested_position": 0.0,
  "wang_mou_analysis": "軍師王謀五層思考法定性診斷摘要（200字以內）",
  "llm_fallback": false
}
✅ Scope & Acceptance Criteria (DoD)
Context Pruning & Prompt 精簡：

Given 傳入當日診斷數據 (isd_triggered, benchmark_roi, champion_agent_roi, today_decision)
When 構建 Gemini API Prompt 時
Then 上下文僅包含此 4 項必要數據，剔除其他 7 個非冠軍 Agent 的歷史歷程；控制 Prompt 於極簡長度，並要求 wang_mou_analysis 摘要限定於 200 繁體中文字內。
ISD 硬性風控與一票否決 (ISD Hard Override)：

Given 當日 isd_triggered == True
When llm_agent.py 處理定性裁決時
Then 無論 API 回覆內容為何，Python 程式碼層強制將 action 覆蓋為 "觀望"、suggested_position 強制鎖定為 0.0，並於 wang_mou_analysis 明確標記觸發 ISD 風控熔斷。
10 秒 Timeout 與 Fail-Safe 靜態降級 (API Breakout)：

Given Gemini API 請求超過 10 秒、遭遇 4xx/5xx 報錯、未設定 GEMINI_API_KEY 或網路中斷
When 呼叫 llm_agent.py 時
Then 靜態捕捉 Exception，絕不拋出錯誤崩潰，立即回傳預設保底數據：
action: "觀望"
suggested_position: 0.0
wang_mou_analysis: "API 連線異常，啟動防禦性降級，建議維持觀望。"
llm_fallback: true
主管線一鍵自動化閉環 (src/main.py)：

Given 執行命令 python src/main.py
When 主管線運作時
Then 依序完成 Step 1 (特徵工程) -> Step 2 (模型重訓) -> Step 3 (當日推估/風控) -> Step 4 (Agent 沙盒模擬) -> Step 5 (LLM 定性裁決)，寫入 data/llm_agent_report.json 並於 Terminal 輸出 JSON，流程 100% 自動化閉環貫穿。
單元與整合測試綠燈 (Test Verification)：

Given 執行 pytest tests/test_llm_agent.py tests/test_main.py
When 包含 Mock API 成功、API Timeout (>10s) 與 ISD 熔斷情境時
Then 單元測試與整合測試 100% 綠燈通過。
🛡️ Edge Cases & Fallback
[GEMINI_API_KEY 缺失]：若環境變數未包含 API Key，llm_agent 必須自動觸發 Fail-Safe 降級，回傳 "llm_fallback": true，不中斷 main.py 執行。
[LLM 回傳非標準 JSON / Markdown 雜訊]：若 Gemini 回傳包含 ```json 標籤或格式破損，需進行安全修剪 Parse；Parse 失敗時自動觸發 Fallback 靜態降級。
❌ Explicit Non-Goals
❌ 嚴禁修改 upstream 核心模組：嚴禁改動 src/data_preprocessing.py、src/train_model.py、src/predict_today.py 與 src/simulate_agents.py。
❌ 嚴禁無 Timeout 或無 try-except 的外部 API 呼叫：Timeout 必須嚴格限定為 10 秒。
❌ 嚴禁冗長診斷輸出：wang_mou_analysis 嚴禁超過 200 繁體中文字。
🛑 Execution Rules For Cline
最少改動：僅修改達成 DoD 所需的最少程式碼，嚴禁順手重構無關檔案或調整抽象層。
爭議即停 (Halt Immediately)：若執行過程需要修改清單外的檔案、發現欄位缺失或遇到未定義的邏輯衝突，必須立即停止動作並擲出報錯，嚴禁自行猜測與通靈實作。