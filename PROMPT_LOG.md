# 构建过程中的 AI 提示词与问题解决记录

本文件记录 min-agent 从零搭建时提出的关键提示词（prompt），以及每个
"踩坑 → 验证 → 修复"的过程。目的是呈现**过程**而不只是结果。

## 0. 需求拆解提示词

> 用一句话说明任务是：实现一个最简 Agent，要求"基础循环 + 工具 + 会话 +
> 上下文管理 + 错误处理 + 测试 + 真实 LLM + 记忆时序说明"。请给出：验收清单、
> 允许/禁止的范围、可选技术方案、以及每一步的产出物。

答复收敛为：禁止使用任何 agent 框架；用 Anthropic 兼容协议直连；四步循环；
本地目录化会话；压缩保"工具对"；三层记忆。

## 1. 技术决策提示词（四连问）

- "用哪个库调 LLM？" → 选 anthropic python SDK（已验证对话流），保留 httpx/openai 备用。
- "多窗口怎么并发？" → 每窗口一个进程 + 每窗口独立会话目录；会话存储用目录而非全局内存。
  （此处更正：最初写的是"单进程多线程"，但实现里 `__main__` 没有任何线程，REPL 的
`input()` 和 `run_turn()` 都在主线程串行执行；进程内唯一的并发是同一回合的并行工具
   调用 `ThreadPoolExecutor`（`loop.py:309`）。此处更正：决策记录与实现不符，以代码为准。）
   —— 后续又补了一条硬约束：`SessionStore.open` 对 `<session>/.lock` 取 OS 排它锁，
   "每窗口一进程"从约定变成强制的，第二进程会被 `LockBusy` 拒绝（CLI 退出码 2），
   解决两个进程各留下一份内存状态、后写者把对方对话整段抹掉的问题（`paths.py` 的
   `FileLock` + 本文件 2.12）。
- "记忆结构？" → 三层（工作/情景/语义），语义层即 facts.json。
- "放弃哪些？" → 不做真正的网络搜索、不做向量库、不做流式 UI。

## 2. 关键探针及其结论（问题解决记录）

### 2.1 thinking 块能否从会话中剔除

**问题**：模型每回合都返回 `thinking` 块；直接原样持久化既占空间，重放时某些
网关对 `signature` 校验敏感。

**探针 prompt**：*"用一个会返回 thinking 块的模型跑一个工具调用回合，然后把
thinking 块从 assistant 消息里删掉，再次请求，观察是否被网关拒绝。再用
parallel tool_use 重复一次。"*

**结论**：删除后照常工作；`tool_result` 往返正常，`end_turn` 结束。→ 转录本只保留
`text + tool_use`，thinking 留给 trace。

### 2.2 并行工具调用

**探针 prompt**：*"同时返回两个 tool_use，观察工具是否都执行、结果是否按
tool_use_id 对应回填。"*

**结论**：支持，且结果顺序不必与调用顺序相同 → 以 `tool_use_id` 强匹配。

### 2.3 tool_use / tool_result 必须成对

**探针 prompt**：*"构造一个只有 tool_use 没有 tool_result 的请求，观察网关是否拒绝。"*

**结论**：被拒。→ 压缩器与修复逻辑都必须"成对移动"，参见 context.py 与
session.py 的 `_drop_hanging_tool_refs`。

### 2.4 temperature 参数在 anthropic SDK 1.x 已不存在

**探针 prompt**：*"真实 API 第一个请求就抛
`Messages.create() got an unexpected keyword argument 'temperature'`。"*

**验证**：`inspect.signature(Anthropic.Messages.create)` 显示该版 SDK 已移除
`temperature`。

**修复**：`AnthropicLLM.complete` 不再转发 `temperature`（意图保留在 LLMRequest 上）。

### 2.5 千分位正则差点毁了 `min(3,7,1)`

**探针 prompt**：*"calculator 支持参数，表达式里如何把 `1,234`

**坑**：先把千分位正则写成 `(?<=\d),(?=\d)`，带参数的调用如 `f(3,7,1)` 会被误拆。

**修复**：改为 `(?<=\d),(?=\d{3}\b)`，只拆"后跟恰好三位再词边界"的逗号。

### 2.6 `date` 命名遮蔽

**问题**：weather 工具需要日期类型，`from datetime import date` 会把同名参数
`date` 遮蔽，导致 schema 推导错误。

**修复**：`from datetime import date as Date`（tools/weather.py）。

### 2.7 摘要折叠进锚点，保持交替

**问题**：压缩后如果插入一条新的独立 user 消息，会破坏 `[user, assistant, user, ...]`
交替被网关拒绝（探针 2.3 已证）。

**修复**：摘要文本合并进首条 user 锚点消息，参见 context.py `compact_messages`。

### 2.8 重复调用护栏的语义

**问题**：最初用 `count > MAX_REPEAT_CALL`，导致需要"第 4 次重复"才触发、浪费一轮。

**修复**：`>=`，同一签名出现 MAX_REPEAT_CALL 次即收尾（loop.py）。

### 2.9 测试里"压缩偷走假模型的剧本"

**现象**：scripted 模型测试忽然"剧本耗尽"。根因：假输入时上下文估算值含工具
schema 开销，触发了压缩器，而压缩器又调用了一次"模型"。

**验证&修复**：把基础设施测试的 CONTEXT_BUDGET 开到极大；只有专门测压缩的用例
才设小预算并显式关闭摘要器。

### 2.10 悬空 tool_use 修复把纯文字尾巴也删了

**问题**：`_drop_hanging_tool_refs` 把"最后一条 assistant 但不是 tool_use"的普通
回答也当成悬挂帧删掉。

**修复**：只有当尾巴里确实含 `tool_use` 块才删除（session.py）。

### 2.11 "数值工具输出"门控误判一位小数

**问题**：`37.5`（一位小数）漏过数值正则
`[\d]{2,}[.,][\d]{2,}`，被当成长期事实存了下去。

**修复**：放宽为 `\d+[.,]\d+`（memory.py `_gate`）。

### 2.12 "多窗口并发"原来是静默丢数据，不是并发

**问题**：每个窗口都按需 `open` 同一会话目录，两个进程各加载一份 transcript
副本、各 append 自己的回合、各把副本整文件写回去——后写者把先写者的回合
无声删光。没有任何文件被写坏，所以原子写救不了，`{ok}` 也不会报错。

**修复**：`SessionStore.open` 对 `<session>/.lock` 取 OS 排它锁（`FileLock`，
Windows `LockFile` / POSIX `flock`），锁的生命周期=窗口生命周期。第二进程
`open` 同一个窗口抛 `LockBusy`，CLI 打印一行（含占锁 pid）并以退出码 2 结束。
锁是内核句柄，进程被 `kill -9` 也会自动释放，不会留下"锁死"状态。测试里为了
模拟双进程开了真的两个 subprocess、还有一个硬杀持有者的用例（tests/test_cli.py、
tests/test_paths.py）。多窗口（同用户不同 session）各自拿各自的锁，继续并行。

### 2.13 facts.json 每回合全量重写

**问题**：`recall()` 在每个用户回合开始都会跑，为了给命中条目的 `hits += 1`
就调用 `save()` 把整个 facts.json 重写一遍——每回合关键路径上多一次 O(n) 落盘。

**修复**：读路径不再写盘。`recall()` 只置 `_dirty`，窗口关闭时由
`Agent.close()` → `MemoryStore.flush()` 一次性落盘（崩了只丢命中计数）。
`remember()` 仍立即写。`save()` 顺带改成"读-改-写"：按 id 与磁盘现状合并，
两个窗口先后存事实不再互相抹除；`_load` 容错（文件损坏/缺字段/未知字段行
跳过不炸）也一并补上（memory.py）。

## 3. 交付前审计与修复

构建期结束、测试全绿之后，仍按题目要求逐条核对，并对全仓（包括测试自身）做逐行
自审。原则：每个问题必须给出文件:行号、复现方式、根因，以及"为什么现有测试没抓到"
——最后一项决定修复是否可信。

-   **中文本地化（textutil / read_docs）**：`raw_keywords`（textutil.py:69）用
    `len>1` 过滤，CJK 单字全扔 → 搜"记忆"恒 0 命中、英文却有命中；`tokenize`
    （textutil.py:23）声明 2-gram 实为 1-gram，真 2-gram 是死代码。此前只有
    "传对了参数"的测试，没有"真搜出内容"的测试。→ 修复 + `tests/test_textutil.py`、
    `tests/test_read_docs.py`。
-   **状态与转录健壮性**：`_wrap_up`（loop.py:318）催场消息漏 `count_turn=False`
    （前两处已修），护栏触发时轮数虚高；`session.py:102` 裸 `json.loads`，转录尾部
    半行会让整个窗口打不开，与自家 docstring 承诺相悖。→ 补参数、坏行逐条跳过。
-   **trace 与流式边界**：机械压缩在控制台渲染成 `None -> None`（trace.py:223 渲染
    分支没跟上）；流式中途异常未分类，SDK 原异常穿透 `run_turn` 的 except 崩 REPL。
    → 渲染补文案；`_complete_or_stream`（loop.py:239）在循环边界统一转 `LLMError`。
-   **测试与打包自查**：`tests/test_tools.py` 同名函数静默覆盖，pytest 不告警；
    `todo.py:178` 注解引用未导入的 `Any`；`weather.py` import `httpx` 却未在
    pyproject 声明。→ 删副本、删死注解；新增 `tests/test_deps.py`——扫描全仓
    import，凡直接引入的三方包必须声明（也是"无框架依赖"的可执行举证）。
-   **二梯队**：压缩预算口径统一（含 tools）；护栏输入确定性追加；`context.py:300`
    机械淘汰补镜像配对；`registry.py:106` hint 只列缺参字段名；`base.py:125`
    Union/Literal 支持 `anyOf`/`enum`、`Optional` 解包；删死代码（`rollback_turn`、
    `forecast_range`、`AgentAborted`）；`set_summary` 改原子写；trace 读写统一复用
    `trace_path`。
-   **完成态**：离线 210 + live 6 全绿；缺陷 → 修复 → 钉死测试 → commit 完整映射
    见 `docs/REFACTOR.md`。

## 4. 组装过程提示词（要点）

- "按上面的决策给出模块划分与文件清单，包含每个文件的职责、公开函数签名。"
  → 产出本文档第 0~2 节依赖的骨架。
- "为 loop / session / context / memory 各写测试，遵守：离线测试用 ScriptedLLM，
  联网测试标 @pytest.mark.live 且默认不跑。" → 91 离线 + 5 live。
  （构建当时；现已 210 离线 + 6 live，数字随缺陷修复新增的用例增长。）
- "最后的演示脚本要有代表性：一句问候、一次计算、一条待办、一次天气。"

## 5. "从零实现"的举证

题目禁止依赖现成 agent 框架。举证方式是一个可执行的自检
（`tests/test_deps.py`）：扫描 `min_agent/**` 全部 import，与 `pyproject.toml`
的 `dependencies` 比对，要求"直接 import 的三方包必须声明"。

当前结果：全包 22 个模块的直接依赖只有 4 个三方包——

-   `anthropic`：仅作 Anthropic Messages 协议的 HTTP 客户端与类型层；
-   `python-dotenv`：读 `.env`；
-   `jsonschema`：校验工具入参；
-   `httpx`：`WEATHER_BACKEND=wttr.in` 的天气请求（3.8 里此前未声明，属于审计揪出的打包事故）。

其余全部是标准库。四步循环、工具注册与 schema 推导、输出解析、会话目录与
OS 排他锁、上下文压缩、三层记忆、trace 全部自研，没有 langgraph / openhands /
openclaw / PI 的任何依赖、导入或变通写法。SDK 只当协议层用，runtime 是手写的。

## 6. 与需求逐条对照

| 需求 | 落地位置 |
|---|---|
| 基本循环 4 步 | loop.py:114 `run_turn` |
| ≥3 个工具 + 注册 | tools/（6 个），schema 从类型注解推导 |
| 输出解析 | parser.py（think/text/tool_use/文本兜底） |
| 多窗口会话 | store.py + session.py，每窗口一目录 |
| 上下文管理：轮次/记忆/追问 | config（MAX_TURNS/MAX_REPEAT_CALL）+ transcript + 追问两类测试 |
| 上下文压缩 | context.py（摘要 + 机械淘汰双路径，超出"基础压缩"） |
| 错误处理 | errors.py 分级 + 工具错误回填 + LLM 重试 |
| 工具调用日志 | trace.py（JSONL + 控制台） + `min-agent trace` 子命令 |
| 测试 | tests/（离线 210 + live 6，`test_deps.py` 含"无框架依赖"自检） |
| 真实 LLM | llm.py（DashScope Anthropic 兼容端点，可切官方） |
| GitHub | README + 本文件 + git 历史 |
| 记忆时序说明 | README「记忆」小节 + docs/architecture.md |
| 缺陷审计过程 | 本文件第 3 节 + docs/REFACTOR.md 变更表 |