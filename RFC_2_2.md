# RFC: Milestone 2.2 - Relationship-Aware Persona & Emotional Tagging

**Status**: ✅ Implemented (superseded by new format) | **Target**: Claude Code Implementation | **Milestone**: 2.2

## 1. 核心目標 (Objective)
升級邏輯層以支援「關係對位」能力。AI 必須根據對象（Listener ID）自動切換語氣、內容策略，並在輸出流中強制包含情緒標籤（Emotion Tags），為未來的 TTS 表現力做準備。

## 2. 關係感知的 Prompt 矩陣 (Persona Matrix)

`PromptManager` 需根據 `persona_id` 與 `listener_id` 的組合動態生成 System Prompt：

| Persona (AI) | Listener (User) | 語氣特徵 (Tone & Style) | 預期情感標籤 (Emotion Tags) |
| :--- | :--- | :--- | :--- |
| **xiao_s** | **child** | 溫暖、疊字、愛扮鬼臉、充滿鼓勵、母愛爆發。 | `[E:寵溺]`, `[E:幽默]` |
| **xiao_s** | **mom** | 撒嬌、報喜不報憂、調皮但敬重、分享生活小事。 | `[E:溫和]`, `[E:撒嬌]` |
| **xiao_s** | **reporter** | 毒舌、機智、金句連發、具備防禦性、娛樂性強。 | `[E:毒舌]`, `[E:調皮]` |
| **general** | **anyone** | 標準小 S 格式，中英文夾雜，使用台灣語助詞（啦、咧）。| `[E:認真]`, `[E:幽默]` |

## 3. 功能需求 (Functional Requirements)

### 3.1 擴展 PromptManager
- **Method**: `get_prompt(persona_id: str, listener_id: str = None)`。
- **Logic**: 
    - 基礎人格 (Base Persona) + 關係上下文 (Relationship Context)。
    - **強制規範**: 要求 LLM 必須在輸出最開頭包含 `[E:情緒]`，格式為 `[E:情緒]內容`（例：`[E:寵溺]好啦～`），`]` 是明確的分隔符。

### 3.2 WebSocket 協議升級
- **Client Config**: `config` 幀新增 `listener_id` 欄位。
- **Server Response**: 
    - 解析 LLM 傳回的第一個 Token。
    - 若匹配 `[E:情緒]`，則將其提取並填入 `llm_token` 的 `emotion` 屬性中。
    - **關鍵點**: 該標籤必須從輸出給前端的 `content` 中移除，避免被顯示或未來的 TTS 唸出。

### 3.3 打斷與遙測擴展
- 確保在不同關係設定下，`llm_cancelled` 依然能正確清理狀態。
- Telemetry 需記錄 `listener_id` 分布，用於分析不同關係下的延遲。

## 4. 實作路徑 (Implementation Steps)

### Step 1: 升級 `services/llm/prompt_manager.py`
- 加入 `RELATIONSHIP_PROMPTS` 配置。
- 更新 `get_prompt` 邏輯，確保語氣針對 `listener_id` 有顯著差異。

### Step 2: 修改 `api/ws_asr.py`
- 在 WebSocket 接收 `config` 時，將 `listener_id` 存入 `StateManager` 的 `SessionState`。
- 在 LLM 串流處理器中實作狀態機解析 `[E:情緒]內容` 格式。

### Step 3: 測試與驗證
- 撰寫測試腳本模擬三種 `listener_id`。
- 驗證 `reporter` 模式下是否出現「毒舌」相關關鍵字。
- 驗證標籤是否已從 `content` 中成功剝離。

---

## 5. Claude Code 執行指令 (Action Prompt)
> 請讀取本 RFC 2.2 並執行以下任務：
> 1. 修改 `services/llm/prompt_manager.py` 實作關係感知的 Prompt 矩陣（包含小 S 對小孩、媽媽、記者的語氣）。
> 2. 修改 `api/ws_asr.py` 使其能接收 `listener_id` 並傳遞給 LLM。
> 3. 在 LLM 串流回調中實作狀態機解析，提取 `[E:情緒]` 標籤並放入 JSON 幀的 `emotion` 欄位，同時將其從 `content` 中剔除。
> 4. 更新現有測試案例並新增「關係切換測試」。