| Phase | Scope | Difficulty | Why |
|---|---|---|---|
| Stage 0 | Contract lock (API/DB/backward-compat) | Medium | Alignment-heavy, low code, high impact decisions |
| Stage 1 | Level 2 foundation (`call_messages`, conversation ASI + evaluator) | Medium | Mostly isolated backend modules, manageable complexity |
| Stage 2 | Level 2 runner + API routing + 40-case suite | Medium-High | Concurrency/report integration + schema compatibility |
| Stage 3 | Level 2 frontend (Conversation tab + metrics UI) | Medium | UI + API integration, moderate product complexity |
| Stage 4 | Level 3 core (`traces`, sandbox executor, trajectory metrics, `call_with_tools`) | High | New abstractions + provider-specific tool-call formats |
| Stage 5 | Level 3 runner + trace persistence + `/traces` API | Very High | Orchestration loops, fault injection, storage, reliability edge cases |
| Stage 6 | Level 3 frontend (Agent tab + timeline + traces page) | High | Rich visualization + async state + data volume handling |
| Stage 7 | Mode 2 SDK (`TraceCollector`) + docs/integration tests | Medium | API ergonomics + correctness, less infra complexity |

Difficulty order (hardest to easiest): **Stage 5 > Stage 4 ≈ Stage 6 > Stage 2 > Stage 1 ≈ Stage 3 ≈ Stage 7 > Stage 0**.
