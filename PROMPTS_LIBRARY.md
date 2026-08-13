# 📊 AI 股市預測與決策引擎系統 - 專案提示詞庫 (Prompts Library)

歡迎使用本專案的專屬提示詞庫！本庫專門收錄並記錄本專案「AI 股市預測與決策引擎系統」中所有與 AI/LLM 互動的提示詞（Prompts）。透過將提示詞模組化，您可以極其方便地進行提示詞版本管理、快速複製，或修改個別提示詞。

---

## 📂 提示詞庫目錄結構

提示詞統一管理於專案根目錄的 `prompts/` 資料夾下：

```text
taiwan_stock_engine/
├── PROMPTS_LIBRARY.md              # 📖 本引導導覽文件 (Root)
└── prompts/                        # 📂 提示詞存放目錄
    ├── wang_mou_system_prompt.txt   # 🧬 首席股市軍師「王謀」系統角色設定與思考框架
    ├── wang_mou_user_prompt.txt     # 📈 「王謀」每日盤後量化數據推理輸入範本
    ├── claude_switch_refactor.txt   # 🛠️ Claude Code / CC Switch 專用代碼重構提示詞
    ├── claude_switch_debug.txt      # 🐞 Claude Code / CC Switch 專用 Debug/故障排除提示詞
    └── gemini_sandbox_analyst.txt   # 🏆 雲端 Gemini 沙盒競技場回測結果分析與戰略優化提示詞
```

---

## 🛡️ 雙 AI 引擎「API Error 400 防護與 Prompt 撰寫鐵律」

當您使用 **CC Switch** 調度 OpenRouter 轉接 DeepSeek 引擎，並搭配 **Claude Code CLI** 時，為了避免 Claude 發起原生 Tool Call（如 `Read` / `Write` / `Edit`）導致 OpenRouter 代理報錯（`API Error: 400 Invalid Anthropic Messages API request`），所有發送給 Claude CLI 且預期只獲得純文字回覆的 Prompt，**開頭必須無條件強制加上防護標頭**：

> **`【請以純文字回答，嚴禁呼叫 Read / Write / Edit 或任何 Tool Call】`**

*本庫中的 `claude_switch_refactor.txt` 與 `claude_switch_debug.txt` 已預先嵌入此防護標頭，請放心複製使用！*
*💡 操作小技巧：在將 Prompt 貼到 Claude Code 視窗前，請務必點擊移除輸入框底部的自動檔案標籤（如 `[simulate_agents.py]`），保持 Context 的乾淨。*

---

## 📖 提示詞詳細說明與使用指南

### 1. 首席股市軍師「王謀」
* **系統提示詞檔**：`prompts/wang_mou_system_prompt.txt`
* **使用者數據檔**：`prompts/wang_mou_user_prompt.txt`
* **定位**：極端的「攻擊型」智能體，負責解讀市場情報，絕對不計代價尋找獲利的最優解。
* **核心理論框架**：**「五層思考法推演」**（技術線型 ➔ 籌碼動力 ➔ 產業/基本面 ➔ 主力心理與假摔 ➔ 終極戰略）。
* **使用方式**：
  * 已整合於 `src/llm_agent.py`，啟動 `ask_wang_mou(...)` 時會自動調用。
  * 亦可手動複製 `wang_mou_system_prompt.txt` 作為系統指令，並將當日盤後量化數據填入 `wang_mou_user_prompt.txt` 範本中，在 Google AI Studio 中與 Gemini 3.6 Flash 對話。

### 2. CC Switch 特種代碼重構 (Refactor)
* **提示詞檔案**：`prompts/claude_switch_refactor.txt`
* **定位**：高階演算法重構與代碼優化。
* **使用方式**：
  1. 打開 `prompts/claude_switch_refactor.txt`，複製完整內容。
  2. 將最下方 `[在此處詳細貼上您要重構的目標代碼、類別或函式需求]` 替換為實際需求。
  3. 貼給 CC Switch/Claude Code，它會以極其優雅且符合本專案核心理論（如 V 轉、假摔演算法）的形式進行代碼優化。

### 3. CC Switch 精準故障排除 (Debug)
* **提示詞檔案**：`prompts/claude_switch_debug.txt`
* **定位**：專門用於解決程式碼崩潰、Exception、數據維度不對齊等 Bug。
* **使用方式**：
  1. 複製 `prompts/claude_switch_debug.txt` 的內容。
  2. 貼上 Terminal 中的錯誤 Traceback 與相關代碼段。
  3. 貼給 CC Switch/Claude Code，快速獲得無 Tool-Call 衝突的 Root Cause 分析與修正代碼。

### 4. 雲端 Gemini 沙盒戰術分析師 (Sandbox Analyst)
* **提示詞檔案**：`prompts/gemini_sandbox_analyst.txt`
* **定位**：宏觀策略導師，負責點評 `simulate_agents.py` 跑出來的 8 大 Agent 沙盒回測報告。
* **使用方式**：
  1. 跑完沙盒模擬後，複製終端機輸出的 ROI、勝率與各 Agent 的表現數據。
  2. 將數據貼入 `prompts/gemini_sandbox_analyst.txt` 中指定的區塊。
  3. 將 Prompt 發送給雲端 Gemini，獲取 R2R 錯題本反思、相似度指數 (GSI) 及信任指數 (RI) 的下一步代碼落地提案。

---

## 📈 提示詞庫維護與變更紀錄 (Changelog)

| 版本 | 日期 | 異動內容描述 | 維護人 |
| :--- | :--- | :--- | :--- |
| **V1.0** | 2026-07-29 | 首發提示詞庫上線！模組化拆分「王謀軍師提示詞」，並新增 API Error 400 防護版 CC Switch 重構、除錯提示詞，與沙盒戰報分析提示詞。 | Cline / Memory Adjutant |
