<p align="center">
  <img src="../frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

<p align="center">
  <strong><a href="../README.md">English</a> | <a href="README_zh-hans.md">简体中文</a> | <a href="README_zh-hant.md">繁體中文</a> | 한국어 | <a href="README_es.md">Español</a> | <a href="README_ja.md">日本語</a> | <a href="README_ru.md">Русский</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_it.md">Italiano</a></strong>
</p>

# ML Intern

문서, 논문, 데이터셋, 클라우드 컴퓨팅 자원에 깊이 접근하면서 Hugging Face 생태계를 활용해 머신러닝 관련 고품질 코드를 자율적으로 조사하고 작성하고 전달하는 ML 인턴입니다.

## 빠른 시작

### 설치

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

#### 이것으로 끝입니다. 이제 `ml-intern` 은 어느 디렉터리에서든 사용할 수 있습니다:

```bash
ml-intern
```

프로젝트 루트에 `.env` 파일을 만들거나, 아래 환경 변수를 셸에서 export 하세요:

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key> # if using anthropic models
OPENAI_API_KEY=<your-openai-api-key> # if using openai models
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token>
```

`HF_TOKEN` 이 설정되어 있지 않으면 CLI 가 첫 실행 시 token 입력을 요청합니다. `GITHUB_TOKEN` 을 얻는 방법은 [여기](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)를 참고하세요.

### 사용 방법

**대화형 모드** (채팅 세션 시작):

```bash
ml-intern
```

**헤드리스 모드** (단일 프롬프트, 자동 승인):

```bash
ml-intern "fine-tune llama on my dataset"
```

**옵션:**

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --model openai/gpt-5.5 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

## 아키텍처

### 구성 요소 개요

```
┌─────────────────────────────────────────────────────────────┐
│                         User/CLI                            │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (170k)   │  │  │  │  │  │
│  │  │  │  │ • Session upload to HF     │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ HF docs & research     │  │  │  │  │  │
│  │  │  │  │  ├─ HF repos, datasets,    │  │  │  │  │  │
│  │  │  │  │  │  jobs, papers           │  │  │  │  │  │
│  │  │  │  │  ├─ GitHub code search     │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  ├─ Planning               │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sandbox, destructive ops) │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### 에이전트 루프 흐름

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## 이벤트

에이전트는 `event_queue` 를 통해 다음 이벤트를 발생시킵니다:

- `processing` - 사용자 입력 처리 시작
- `ready` - 에이전트가 입력을 받을 준비가 됨
- `assistant_chunk` - 스트리밍 토큰 조각
- `assistant_message` - LLM 의 전체 응답 텍스트
- `assistant_stream_end` - 토큰 스트림 종료
- `tool_call` - 인자와 함께 도구 호출
- `tool_output` - 도구 실행 결과
- `tool_log` - 도구의 정보성 로그 메시지
- `tool_state_change` - 도구 실행 중 상태 변경
- `approval_required` - 민감한 작업에 대한 사용자 승인 요청
- `turn_complete` - 에이전트가 현재 턴 처리를 완료함
- `error` - 처리 중 오류 발생
- `interrupted` - 에이전트가 중단됨
- `compacted` - 컨텍스트가 압축됨
- `undo_complete` - 실행 취소 작업 완료
- `shutdown` - 에이전트 종료 중

## 개발

### 내장 도구 추가

`agent/core/tools.py` 를 수정하세요:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### MCP 서버 추가

CLI 기본값은 `configs/cli_agent_config.json`, 웹 세션 기본값은 `configs/frontend_agent_config.json` 을 수정하세요:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

참고: `${YOUR_TOKEN}` 같은 환경 변수는 `.env` 에서 자동으로 치환됩니다。
