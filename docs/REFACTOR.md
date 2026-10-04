# min-agent 修复与工程化记录（REFACTOR）

本文件记录一次集中审计+修复的完整过程：每个修复对应哪个缺陷、哪个回归测试、
哪个 commit，以及据此沉淀出来的契约与维护须知。**代码变动先于本文档**，本文是
修复后的"真相支持"，不是未来的计划。

## 1. 心智模型（读这条就够）

```
                     ┌─────────────────────────────────────────┐
                     │              终端 / REPL                 │
                     │  __main__.py → 命令解析 → main()          │
                     └────────────────────┬────────────────────┘
                                          │ 用户输入
                                          v
   ┌─────────────── 一回合（四步循环，loop.py::Agent._ask/_run_tools）───────────────┐
   │ 1 收到输入 ── session.append(user, count_turn=True) → transcript.jsonl        │
   │ 2 决定     ── llm.complete/stream（LLMRequest 不再携带 temperature）            │
   │ 3 行动     ── tool registry.call（并行 + 超时，失败是数据不是异常）               │
   │ 4 继续/结束 ─ 有 tool_use 回到 2；否则 final → trace（前缀只打一次）             │
   └─┬─────────────────────────────────────────────────────────────┬──────────────┘
     │ 每个 user 回合开始前                                │ 上下文超过预算时
     v                                                     v
   memory.recall ──► 注入系统提示词                        context.compact
   （只读，不落盘）   Long-term memory 段                   ─ 摘要折叠进锚点 user 消息
     │                                                       ─ 机械清除"最早的完整回合"
     v                                                       ─ 事件进 trace(kind=compact)
   .memory/<user>/facts.json（remember/extract_facts 写）

   横切：trace（JSONL+ansi，kind 白名单强制）
         safe_segment（所有用户可见路径的单一净化源）
         FileLock（<session>/.lock，独占，被占时退出码 2）
```

三个关键不变式：

1. **记忆是上下文，不是指令** —— 召回结果永远进系统提示词，绝不混进 user 消息。
2. **工具失败是数据** —— `{ok:false, error}` 还给模型，让模型决定重试/解释；
   只有超时/网络/编码级错误才抛异常。
3. **文件损坏一律降级** —— facts/meta/todo/sessions 枚举，读不动就跳过或给默认值，
   绝不把整个窗口带走。

## 2. 修复 → 回归测试 → commit 映射

| 缺陷/改动 | 症状 | 回归测试 | commit |
|---|---|---|---|
| FileLock 追加式覆写 | 锁被接管后 owner 变成多字节拼接断言 | `test_owner_stamp_overwrites_in_place_not_append`、`test_busy_message_names_the_current_holder_not_the_first` | `17d3b66` |
| 记忆门控旧规则 | 文档示例被拒 / 数值型工具输出误拒 | `test_every_documented_example_passes_the_gate`、`test_numeric_facts_are_stored_not_mistaken_for_tool_output` | `6fbfea6` |
| extract_facts 混入助手消息 | 抽取时把 AI 自己的话当用户事实 | `test_extract_facts_reads_only_the_users_turns` | `15b94fc` |
| tail 计算了却没进 prompt | 摘要只看到"0 条"用户消息 | 同上用例断言 prompt 含 tail | `15b94fc` |
| 搜索"伪 BM25"+snippet 偏置 | 排名靠词频而非相关性，snippet 不居中 | `test_results_are_ranked_real_bm25`、`test_snippet_is_centered_on_the_hit`、`..._ellipsis_flags_truncation` | `01a74e5` |
| search 子模块被遮蔽 | `from ..tools import search` 拿到 ToolSpec 而非模块 | `test_package_search_name_is_the_module_not_the_tool` | `01a74e5` |
| 容器/联合类型 schema 非法 | 工具带 list/tuple/Annotated 时注册即炸 | `test_container_types_map_to_valid_json_schema`、`test_broken_schema_surfaces_as_value_error_not_input_error` | `f29b29d` |
| 路径净化各写各的 | 一处漏净化的绕过另一处 | `test_windows_reserved_device_names_are_neutralised`、`test_delegating_callers_no_longer_inline_their_own_sanitiser` | `c6d76bb` |
| 催场消息计入 turn 数 | 用户只问 1 次却记 3 轮 | `test_nudges_do_not_count_as_turns` | `2521e58` |
| 重复护栏全局计数 | `A,B,A,B` 被误判循环 | `test_repeat_guard_requires_consecutive_not_global_repeats`、`test_interleaved_tools_do_not_trip_the_guard` | `06cfc70` |
| 机械压缩 note 是死参数 | 驱逐行为在 trace 里无痕 | `test_mechanical_eviction_note_reaches_the_trace` | `21f6a58` |
| 三处 json 加载不宽容 | 坏 meta/todo 一行带崩窗口/列表 | `test_corrupt_meta_degrades_to_defaults`、`test_list_sessions_skips_a_corrupt_meta`、`test_load_skips_a_foreign_row_without_losing_the_rest` | `4df9f66` |
| trace emit 无契约 | kind 拼错只落一条空白 console | `test_unknown_emit_kind_is_rejected` | `1a21d28` |

纯重构（无测试，行为不变）：删 `LLMRequest.temperature` 与死 import（`e4586fe`）、
`store/memory` json import 上提（`aea5ee9`）、魔法数命名（`91f0b80`）。

## 3. 沉淀的契约（改动易踩，测试钉死）

- **trace kinds**：`user / llm_request / llm_response / reasoning / tool_call /
  tool_result / memory_recall / memory_store / compact / error / warning / final`。
  字段定义见 `trace.py` 模块 docstring；`emit` 会对未知 kind 抛 `ValueError`。
- **FileLock owner 槽位**：锁字节在偏移 0，`_OWNER_WIDTH=64`，`_stamp_owner` 用
  `ftruncate`+`seek(1)` 原地覆写 63 字节，绝不追加。
- **safe_segment**：`min_agent/paths.py` 是唯一实现；`loop/session/trace/__main__`
  一律委托；Windows 保留名（aux/con/nul/prn/com1-9/lpt1-9）→ `default`。
- **schema 错误分层**：工具注解坏 → `ValueError("invalid JSON schema")` → `call()`
  呈现为 `internal error`；参数不合法 → `ToolInputError`。两者永不混淆。
- **Session.append(role, content, *, count_turn=True)**：催场消息必须 `count_turn=False`。
- **门控示例**：`_DOCUMENTED_FACTS` 是唯一事实源，文档与其保持同步。

## 4. 扫描结论

- **P3-16 文档漂移**：README 测试数停在 155（现 178）、门控规则残留"数值型工具输出
  拒绝"、handbook 工具表漏 `remember`——本次已修正，并把示例事实指向
  `_DOCUMENTED_FACTS`。
- **P3-18 钉死审计**：上表每项修复都有对应回归测试；无试探性改动需要回退。
- **P3-20 魔法数**：`_THINK_PREVIEW_CHARS`、`_RESULT_PREVIEW_CHARS`、
  `_TAIL_MAX_CHARS`、`_TRACE_LINE_TRUNCATE` 命名化；超时/字符上限等策略值仍统一在
  `config.py`。

## 5. 维护须知

1. 任何 fix 必须带回归测试，`pytest tests/` 必须全绿（live 默认剔除，
   `pytest -m live` 6 项联网另跑）。
2. 被占窗口的进程会让 `min-agent` 退出码 2 —— 冒烟时先清掉两个长期测试窗口。
3. 改 `_DOCUMENTED_FACTS` 的同时，先跑 `test_every_documented_example_passes_the_gate`。
4. 提交按"一项一 commit"；改动列表同步更新本文件。