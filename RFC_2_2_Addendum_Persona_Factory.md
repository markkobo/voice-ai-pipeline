# RFC 2.2 Addendum: Data-Driven Persona Factory

**Status**: Active | **Target**: Claude Code | **Enhancement**: Decoupling Persona Logic

## 1. 結構化變更 (Structural Changes)

為了實現「人格即數據 (Persona as Data)」的願景，新增資源目錄：
- **Path**: `app/resources/personas/`
- **Purpose**: 存放所有數位遺產的配置 JSON，讓系統在不修改代碼的情況下即可載入新角色。

## 2. Persona JSON 規格 (Schema Definition)

每個 Persona 配置檔（例如 `xiao_s.json`）必須符合以下結構：

```json
{
  "persona_id": "string",
  "base_personality": "核心性格描述，包含語言風格與禁忌。",
  "emotion_instruction": "關於如何在輸出中嵌入 [情感: 類型] 的指令。",
  "relationships": {
    "listener_id_1": "對該特定對象的語氣描述",
    "listener_id_2": "對該特定對象的語氣描述"
  },
  "default_relationship": "查無匹配對象時的通用語氣。"
}
3. 功能邏輯微調 (Logic Refinement)
3.1 PromptManager 工廠模式
動態載入: get_prompt(persona_id, listener_id) 會根據 persona_id 前往 app/resources/personas/ 讀取對應的 JSON。

快取機制: (建議) 在內部使用簡單字典快取已讀取的 JSON，避免頻繁磁碟 IO。

合成邏輯:

Prompt = base_personality + relationship_modifier + emotion_instruction

其中 relationship_modifier 由 listener_id 決定。

3.2 系統整合
api/ws_asr.py 需確保傳遞正確的 persona_id（預設 xiao_s）與從 config 幀獲取的 listener_id。

繼續維持 Regex 解析 邏輯，從 LLM 串流中剝離情感標籤。

4. Claude Code 執行指令 (Updated Action Prompt)
「請依照 RFC 2.2 Addendum 執行架構解耦優化：

建立 app/resources/personas/ 目錄，並建立 xiao_s.json 包含其性格與各類關係（小孩、媽媽、記者）的描述。

重構 services/llm/prompt_manager.py 為工廠模式，支援從 JSON 檔案動態載入人格。

修改 api/ws_asr.py 確保 persona_id 與 listener_id 能正確流轉至工廠。

驗證情感標籤解析邏輯（Regex）在動態載入的 Prompt 下依然運作正常。

新增測試案例：建立一個名為 test_legacy.json 的偽造人格，驗證系統能否在不改動代碼的情況下切換到新的人格語氣。」