========= Written in Korean first, then English ==========

======== 한국어 ========

# clx-ultracode

AI 에이전트(Hermes Agent, Claude Code, Codex)를 위한 합의 의사결정 플러그인입니다. 핵심 기능은
**3개 에이전트의 적대적 토론**입니다: 세 에이전트가 근거와 이유로 서로를 설득하고, 더 잘 논증된
주장은 이유와 함께 수용하며, **만장일치에 이르렀을 때에만** 그 결정을 채택합니다. 합의에 이르지
못하면 반대 의견과 함께 정직하게 `no_consensus`를 반환합니다. 수렴 판정은 모델이 아니라 결정론적
코드가 통제합니다.

공개/discovery 이름: `clx-ultracode`. Python 배포·Hermes entry-point 이름은 호환을 위해
`cluxion-agentplugin-effort-ultracode`를 유지합니다.
에이전트가 발견하는 스킬 ID와 경로도 `clx-ultracode` / `skills/clx-ultracode/`입니다.

## 설치

```bash
pip install cluxion-agentplugin-effort-ultracode
```

### Hermes Agent에서 사용

Hermes의 플러그인 설정에 추가한 뒤 Hermes를 재시작하세요.

```yaml
plugins:
  enabled:
    - cluxion-agentplugin-effort-ultracode
```

Hermes를 통해 제공되는 로컬 모델(vLLM/MLX)에서도 동일하게 동작합니다.
`--models` 또는 `CLUXION_EFFORT_ULTRACODE_HERMES_MODEL`로 기본 모델과 다른 모델을 지정할
때만 Hermes의 명시적 신뢰 설정이 필요합니다.
자동으로 활성화하지 말고, 허용할 모델만 설정하세요.

```yaml
plugins:
  entries:
    cluxion-agentplugin-effort-ultracode:
      llm:
        allow_model_override: true
        allowed_models:
          - provider/model-id
```

### Codex CLI에서 사용

로컬 checkout:

```bash
codex plugin marketplace add cluxion-local /path/to/clx-ultracode
codex plugin add clx-ultracode@cluxion-local
```

Git URL:

```bash
codex plugin marketplace add cluxion https://github.com/cluxion/clx-ultracode.git
codex plugin add clx-ultracode@cluxion
```

Codex는 루트 `.codex-plugin/plugin.json`, `commands/`, `skills/`를 읽습니다. `[plugins.<name>] command`
형태의 별도 config snippet은 사용하지 않습니다.

### Claude Code에서 사용

같은 checkout을 Claude Code 플러그인으로 설치하면 루트 `.claude-plugin/plugin.json`, `commands/`,
`skills/`가 사용됩니다. 명령과 스킬은 `cluxion-ultracode` CLI를 호출하고, host agent가 실행과 최종
응답을 소유합니다.

## 사용

Hermes에서는 `cluxion_consensus` 도구로 제공됩니다. CLI로 직접 실행할 수도 있습니다.

```bash
cluxion-ultracode consensus --question "이 제안을 채택할까?" --adapter hermes
cluxion-ultracode consensus --question "이 제안을 채택할까?" --adapter codex
cluxion-ultracode consensus --question "이 제안을 채택할까?" --adapter mock-unanimous
cluxion-ultracode consensus --question-file ./decision.txt --adapter hermes
cluxion-ultracode consensus --question "이 제안을 채택할까?" --rounds 3 --agents 3 --agent-timeout 180 --debate-budget 600 --budget-tokens 120000 --models cheap,strong,cheap
```

`--adapter hermes`(기본값)는 Hermes host의 `ctx.llm` 표면(플러그인 내부) 또는 독립 실행 시
`hermes ultracode-llm` stdin/stdout 브릿지를 사용합니다. Codex CLI host에서는 `--adapter codex`가
권장 backend이며 `codex exec`의 `--output-last-message`로 최종 응답을 캡처합니다.
`--adapter mock-*`는 실제 모델 호출 없이 결정론적 로컬 테스트용입니다.

최대 fan-out은 `agents * (rounds + 1)` 논리 agent/adapter 호출입니다. 예를 들어 기본 3 agents,
3 rounds는 최대 12회 논리 호출입니다. Hermes structured 출력의 첫 응답이 잘못되면 논리 호출마다
provider JSON-repair 호출이 최대 1회 추가될 수 있습니다. `--agent-timeout`은 단일 agent 호출 제한, `--debate-budget`은 전체
토론 시간 예산, `--budget-tokens`는 전체 토큰 ceiling입니다. Backend usage가 있으면 실제 토큰을 쓰고,
없으면 chars/4 estimator로 `estimated: true`를 표시합니다. 긴 질문은 `--question-file PATH` 또는
`--question -`(stdin)로 전달할 수 있습니다. `--models`는 agent seat에 순환 배정됩니다.

저널이 활성화된 CLI/plugin 실행은 replay 순서를 보존하기 위해 agent 호출을 직렬화합니다. agent가 멈추거나
timeout/완료 오류를 내면 현재 실행이 중단되며 quorum에서 제외하고 계속하지 않습니다. 기록된 성공 prefix가
있으면 `--resume <run_id>`가 이를 재과금 없이 replay한 뒤 기록되지 않은 실패 호출부터 다시 시도합니다(다시
과금될 수 있음). 첫 호출이 실패하면 journal이 아직 없으므로 새 run이 필요합니다. Timeout drop과 `MIN_QUORUM`
지속은 비저널 병렬 core 경로에만 적용됩니다.

만장일치면 결정과 근거를, 아니면 반대 의견을 포함한 `no_consensus`를 반환합니다. 예산 초과나 quorum
상실로 중단되면 `status: "aborted"`, `abort_reason`, `rounds_completed`, partial `transcript`를 반환합니다.
검증 실패는 `invalid_question`, `invalid_models`, `invalid_agents`, `invalid_rounds`, `invalid_budget`,
`invalid_timeout` 중 하나를 `error`로 반환하고 기존 `message`를 유지합니다. LLM 호출 전 실패는 저널 파일을
남기지 않으며, 그런 run_id resume은 `journal_not_found`를 반환합니다. 저널 디렉터리는 `0700`, 파일은
`0600`으로 저장됩니다. Resume은 줄바꿈 없이 끊긴 마지막 JSONL 조각만 자동 복구하며, 줄바꿈이 끝난 레코드나
중간 레코드 손상은 바이트를 변경하지 않고 `journal_corrupt`를 반환합니다.

## 점검

설치·Hermes 계약·LLM 백엔드 상태를 결정론적으로 자가 진단합니다. 같은 상태면 항상 같은 결과를
출력하고, 문제가 있으면 증상과 해결 단계를 그대로 알려줍니다.

```bash
cluxion-ultracode doctor          # 사람용 요약
cluxion-ultracode doctor --json   # 구조화 출력
```

Hermes 안에서는 `ultracode_doctor` 도구로도 노출됩니다.

## 슬래시 커맨드 (0.1.16+)

Codex/Claude Code 플러그인 명령:

```
/clx-consensus 이 리팩터링 방향을 채택할까?
/ultracode-doctor
```

Hermes 플러그인 명령:

```
/clx-consensus 이 리팩터링 방향을 채택할까?
/ultracode-doctor
```

Hermes에서는 `/` 입력 시 🔌로 표시 · consensus는 도구 `cluxion_consensus`와 동일.

## 라이선스

Apache-2.0

============ English ==========

# clx-ultracode

A consensus decision plugin for AI agents (Hermes Agent, Claude Code, Codex). Its headline
feature is a **3-agent adversarial debate**: three agents argue from evidence and reasons,
concede points that are better-argued (with a stated reason), and only a **unanimous**
agreement becomes the decision. If they cannot agree, it returns an honest `no_consensus` with
the dissent. Convergence is controlled by deterministic code, not by the model.

Public/discovery name: `clx-ultracode`. The Python distribution and Hermes entry-point name
remain `cluxion-agentplugin-effort-ultracode` for compatibility.
The agent-discovered skill ID and path are also `clx-ultracode` and `skills/clx-ultracode/`.

## Install

```bash
pip install cluxion-agentplugin-effort-ultracode
```

### Use with Hermes Agent

Add it to the Hermes plugin configuration, then restart Hermes:

```yaml
plugins:
  enabled:
    - cluxion-agentplugin-effort-ultracode
```

It works the same with local models (vLLM/MLX) served through Hermes.
Only explicit `--models` or `CLUXION_EFFORT_ULTRACODE_HERMES_MODEL` overrides need Hermes trust
configuration. Do not enable it implicitly; allow only the models you intend to use:

```yaml
plugins:
  entries:
    cluxion-agentplugin-effort-ultracode:
      llm:
        allow_model_override: true
        allowed_models:
          - provider/model-id
```

### Use with Codex CLI

Local checkout:

```bash
codex plugin marketplace add cluxion-local /path/to/clx-ultracode
codex plugin add clx-ultracode@cluxion-local
```

Git URL:

```bash
codex plugin marketplace add cluxion https://github.com/cluxion/clx-ultracode.git
codex plugin add clx-ultracode@cluxion
```

Codex reads the root `.codex-plugin/plugin.json`, `commands/`, and `skills/`. Do not use a
`[plugins.<name>] command` config snippet; Codex plugins are marketplace plugins.

### Use with Claude Code

Install the same checkout as a Claude Code plugin. Claude Code reads the root
`.claude-plugin/plugin.json`, `commands/`, and `skills/`. The commands and skill call the
`cluxion-ultracode` CLI; the host agent owns execution and final answers.

## Use

In Hermes it is available as the `cluxion_consensus` tool. You can also run it from the CLI:

```bash
cluxion-ultracode consensus --question "Should we adopt the proposal?" --adapter hermes
cluxion-ultracode consensus --question "Should we adopt the proposal?" --adapter codex
cluxion-ultracode consensus --question "Should we adopt the proposal?" --adapter mock-unanimous
cluxion-ultracode consensus --question-file ./decision.txt --adapter hermes
cluxion-ultracode consensus --question "Should we adopt the proposal?" --rounds 3 --agents 3 --agent-timeout 180 --debate-budget 600 --budget-tokens 120000 --models cheap,strong,cheap
```

`--adapter hermes` (default) uses the host `ctx.llm` surface inside Hermes, or the standalone
`hermes ultracode-llm` stdin/stdout bridge outside the host. On Codex CLI hosts, `--adapter codex`
is the recommended backend and captures the final answer through `codex exec --output-last-message`.
`--adapter mock-*` runs deterministic local tests without live model calls.

Maximum fan-out is `agents * (rounds + 1)` logical agent/adapter calls. For example, the
default 3 agents and 3 rounds make at most 12 logical calls. If the first Hermes structured
response is malformed, each logical call can add at most one provider JSON-repair call. `--agent-timeout` caps one agent call;
`--debate-budget` caps the whole debate time, and `--budget-tokens` caps total tokens. Token usage
is real when the backend reports it, otherwise chars/4 with `estimated: true`. Long questions can be
passed with `--question-file PATH` or `--question -` (stdin). `--models` cycles models across agent
seats.

Journaled runs (CLI/plugin) serialize agent calls to preserve replay order. If an agent hangs or
returns a timeout/completion error, the current invocation ABORTS — it is not dropped-and-continued.
`--resume <run_id>` replays the recorded successful prefix without re-billing, then retries from the
first unrecorded (failed) call, which may be billed again; if the first call fails there is no journal
yet, so start a fresh run. Timeout-drop and `MIN_QUORUM` continuation apply only to the non-journaled
parallel core path.

On unanimity it returns the decision and rationale; otherwise a `no_consensus` with the dissent.
If budget or quorum aborts the run, it returns `status: "aborted"`, `abort_reason`,
`rounds_completed`, and the partial `transcript`. Validation failures use `invalid_question`,
`invalid_models`, `invalid_agents`, `invalid_rounds`, `invalid_budget`, or `invalid_timeout` as
`error` while keeping the original `message`. Failures before the first LLM call leave no journal
file; resuming that run_id returns `journal_not_found`. Journal directories are `0700` and journal
files are `0600`. Resume repairs only a torn final JSONL fragment without a trailing newline;
newline-terminated or mid-file corruption returns `journal_corrupt` without changing the bytes.

## Diagnostics

A deterministic self-check of install, the Hermes contract, and the LLM backend. The same state
always prints the same result, and on any problem it shows the symptom and the exact fix steps.

```bash
cluxion-ultracode doctor          # human summary
cluxion-ultracode doctor --json   # structured output
```

Also exposed inside Hermes as the `ultracode_doctor` tool.

## Slash commands (0.1.16+)

Codex/Claude Code plugin commands:

```
/clx-consensus Should we adopt this refactor direction?
/ultracode-doctor
```

Hermes plugin commands:

```
/clx-consensus Should we adopt this refactor direction?
/ultracode-doctor
```

In Hermes, shows in `/` autocomplete with 🔌 · consensus matches tool `cluxion_consensus`.

## License

Apache-2.0
