# MEMORY_BANK.md - 台股 AI 預測與決策引擎 (AVM V5.3 Complete)

> **Last Updated:** 2026-08-12  
> **Status:** Phase 1 ~ Phase 5 全數 100% 圓滿功成 (45/45 Unit Tests Passed) | **SYSTEM STATUS:** Production Ready  
> **Architecture Version:** V5.3 (Full Pipeline Automated)  

---

## 🎯 V5.3 全系統終極戰果 (V5.3 Final Achievements)

* **全管線一鍵自動化 (`src/main.py`)**：串聯特徵工程 ➔ 模型重訓 ➔ 當日推估/風控 ➔ 9大 Agent 沙盒 ➔ LLM 定性裁決，5 大步驟 100% 閉環貫穿[cite: 1]。
* **沙盒冠軍績效亮點**：A9 冠軍 Agent (`A9_TOP_DEFENSE_CONSERVATIVE`) 實測 ROI 精確達到 **`+130.61%`**（總交易次數僅 8 筆閉環），徹底碾壓 #0 BUY_HOLD (+40.53%)[cite: 1, 2]。
* **雙層風控硬否決 (Double-Lock Security)**：當日檢測到 $GSI = 3.4558$ 觸發熔斷，Python 程式碼層與 LLM 王謀定性裁決同步執行「一票否決」，強制壓制建議倉位為 `0.0`（觀望）[cite: 1]。
* **10 秒 Timeout 與 API Fail-Safe**：`src/llm_agent.py` 具備 10s 連線斷路器與 Exception 靜態保底，確保主管線 100% 絕不崩潰[cite: 1]。
* **Console Log 100% 純淨無雜訊**：`src/train_model.py` 頂層注入動態路徑防禦，徹底消滅 `ModuleNotFoundError` 警告，極致流暢[cite: 1]。

---

## 1. 專案定位與角色設定 (System Identity & Persona)

### 🤖 Role: Matt Pocock Style AI Engineering Partner
* **核心工程原則：** 消除 "Vibe Coding"，貫徹嚴格的軟體工程紀律與 TDD 驅動開發[cite: 1]。
* **對齊優先 (/grill-me & /grill-with-docs)：** 絕不盲目直接寫 Feature 代碼。任何新功能必須先進行多維度拷問，釐清 Edge Cases、Failure Modes、Architecture Seams 與 Non-Goals[cite: 1]。
* **Deep Module 設計 (/codebase-design)：** 追求介面極簡、內部隱藏複雜度的深層模組，防止程式碼熱力學熵增與泥球架構（Ball of Mud）[cite: 1]。
* **雙軸 Code Review (/code-review)：** 嚴格從 **Spec 軸**（需求忠實度、防杜 Scope Creep）與 **Standards 軸**（設計模式、無 Fowler Code Smells）進行評估[cite: 1]。

---

## 2. 領域語言與關鍵指標 (Ubiquitous Language)

| 術語 / 指標 | 領域意義 | 數學與邏輯定義 |
| :--- | :--- | :--- |
| **$M_{\text{ref}}$** | 長期參考模型 (Golden Baseline) | 採用過去 3 年全量數據，低衰減率 (`0.0005`)，匯出至 `data/model_2379_ref.json`[cite: 1]。 |
| **$M_{\text{conj}}$** | 短期適應模型 (Micro Trend) | 採用近 120 天數據，高衰減率 (`0.0025`)，匯出至 `data/model_2379_conj.json` 與 `model_2379.json`[cite: 1]。 |
| **$RI$ (Reliance Index)** | 信任指數（長短雙腦相似度） | $RI = 1.0 - \|p_{\text{ref}} - p_{\text{conj}}\|$。當雙腦預測分歧過大時觸發 ISD 熔斷[cite: 1]。 |
| **$GSI$ (Global Similarity Index)** | 全域相似度指數（橫斷面黑天鵝） | 馬氏距離 $d = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$，baseline 存至 `data/gsi_baseline.npz`[cite: 1]。 |
| **ISD Circuit Breaker** | 智慧阻斷決策電路 | 當 $DQI_x$ 損壞、$RI < 0.60$ 或 $GSI > 3.00$ 時觸發，強制壓制 `suggestedPosition = 0.0`，當日 9 大 Agent 一律 `"HOLD"`[cite: 1]。 |
| **Settlement Engine** | $T+2$ 交割與擬真費用模型 | 包含 Typical Price 盤中追價撮合、0.33% 摩擦成本（0.3% 滑價 + 手續費/證交稅）與 $T+2$ 資金佇列[cite: 1]。 |
| **High-Confidence Bypass** | 風險分級高勝率旁路 | 勝率達標（Aggressive $0.70$ / Moderate $0.75$ / Conservative $0.80$）且無 ISD 熔斷時，繞過型態過濾器直接 100% 建倉[cite: 1, 2]。 |
| **Three-Track Exit** | 優化三軌離場機制 | 軌道 1（8% 硬停損）、軌道 2（$Z_{\text{BIAS}} \ge 2.5 \land p_{\text{conj}} < 0.60$ 雙重確認）、軌道 3（$p_{\text{conj}} < 0.40$ 趨勢轉弱離場）[cite: 1]。 |

---

## 3. 系統實體模組與開發進度 (System Module Status)

| 模組路徑 | 當前版本 | 驗收狀態 | 核心職責與產出檔 |
| :--- | :--- | :--- | :--- |
| **`src/data_preprocessing.py`** | V3.5 | 🟢 PASSED | 洗淨 18 維籌碼特徵與 $Y_{\text{Alpha}}$ 標籤 $\rightarrow$ `data/features_2379.csv`[cite: 1] |
| **`src/train_model.py`** | V3.5 | 🟢 PASSED | 匯出雙模型與 $GSI$ Ridge 正則化/虛擬逆矩陣（頂層注入動態路徑防禦） $\rightarrow$ `data/model_2379_ref.json`, `model_2379_conj.json`, `gsi_baseline.npz`[cite: 1] |
| **`src/predict_today.py`** | V3.4 | 🟢 PASSED | 雙大腦推估與 ISD 智慧熔斷電路 $\rightarrow$ `data/isd_predict_report.json`[cite: 1] |
| **`src/simulate_agents.py`** | V5.3 | 🟢 PASSED | 9 大 Agent 量化沙盒（風險分級旁路 + 三軌離場，ROI +130.61%, 8 筆交易）$\rightarrow$ `data/simulation_results.json`, `trade_journals.json`[cite: 1, 2] |
| **`src/llm_agent.py`** | V5.3 | 🟢 PASSED | 軍師王謀定性裁決（Gemini 3.6 Flash + Context Pruning + ISD 硬覆蓋 + 10s Fail-Safe）$\rightarrow$ `data/llm_agent_report.json`[cite: 1] |
| **`src/main.py`** | V5.3 | 🟢 PASSED | 全系統五大模組一鍵全自動調度艙 $\rightarrow$ Terminal Console & `data/final_war_room_report.json`[cite: 1] |

---

## 4. 系統防禦與 Failure Modes (Circuit Breakers & Fallbacks)

1. **奇異矩陣降級防禦 (GSI Robustness)**：歷史協方差矩陣加入 Ridge 正則化項 $\Sigma_{\text{reg}} = \Sigma + 10^{-6} \cdot I$，異常時降級至 Standardized Euclidean 距離[cite: 1]。
2. **ISD 熔斷最高優先權 (Hard Circuit Breaker Override)**：`isd_triggered == True` 時擁有絕對最高權重，無視模型勝率與高勝率旁路，強制壓制 `suggested_position = 0.0`[cite: 1]。
3. **LLM API 斷路保護 (API Fail-Safe)**：超過 10s、無 API Key 或連線異常時，靜態捕捉 Exception 並回傳 `"llm_fallback": true` 保底 JSON，確保 `main.py` 順利完結[cite: 1]。
4. **路徑載入潔淨防禦 (Path Injection Defense)**：`src/train_model.py` 最頂層以 `sys.path.insert(0, project_root)` 強制提升尋址優先權，杜絕 `ModuleNotFoundError`[cite: 1]。

---

## 5. 後續里程碑與預定 Roadmap (Pending Roadmap)

- [x] **Phase 1 ~ Phase 5 全數完成 (TICKET-01 ~ TICKET-05 + TICKET-FIX)**：
  - 核心量化引擎、雙大腦特訓、ISD 風控熔斷、V5.3 競技場重構與 LLM 王謀調度艙 100% 綠燈閉環[cite: 1, 2]。
- [ ] **Phase 6A: 實時推播通知模組 (Real-Time Notification Bot)**：
  * 實作 Telegram / Line Bot 串接，自動於每日盤後將王謀定性診斷報告與沙盒冠軍決策推播至使用者手機。
- [ ] **Phase 6B: 自動化排程機制 (Automated Task Scheduler)**：
  * 設置系統層 Cron Job / Task Scheduler，於交易日 14:30 自動觸發 `python src/main.py` 並備份運行日誌。
- [ ] **Phase 6C: 多標的擴展引擎 (Multi-Ticker Engine)**：
  * 將目前單一瑞昱 (2379.TW) 模組泛化，支援台股 Top 50 權值股之平行化分析與量化回測。