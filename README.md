# min-agent

一个**从零实现**的最小 Agent（取件题"Vibe Coding"，无任何现成 agent 框架）。

- ✅ 最简循环：**收到输入 → 决定回答或调用工具 → 执行工具 → 决定继续或结束**
- ✅ 工具注册机制：名称/描述/参数 schema（从类型注解自动推导）
- ✅ 输出解析：thinking / tool_call / final answer 解析 + 文本兜底
- ✅ 多会话：同一用户多个独立窗口，可多进程并行；每个窗口独占一进程，锁拒绝重复打开
- ✅ 上下文管理：最长轮数上限、重复调用护栏、摘要式压缩
- ✅ 错误处理：工具失败是数据、LLM 失败重试、分级异常
- ✅ 运行痕迹：JSONL trace + 控制台渲染
- ✅ 测试：`pytest` 全离线 144 项 + `pytest -m live` 真实 API 6 项
- ✅ 真实 LLM API（DashScope Anthropic 兼容端点）

## 快速开始

```bash
pip install -e .
copy .env.example .env   # 填入 ANTHROPIC_API_KEY（DashScope 或官方）
```

运行：

```bash
min-agent --user alice --session 周末规划   # 打开窗口（REPL）
min-agent --demo                          # 内置演示会话
min-agent --user alice sessions           # 列出 alice 的窗口
min-agent --user alice trace --session 周末规划   # 打印 trace
min-agent --user alice tools              # 列出可用工具
```

窗口内命令：`/help`、`/quit`、`/tools`、`/session`。

测试：

```bash
pytest tests/ -q               # 144 项离线（零网络）
pytest -m live -q              # 6 项真实 API（联网，花钱/算力）
```

## 系统设计

```
min_agent/
  config.py    中心配置（所有可调参数，.env → 一次解析）
  errors.py    错误层级 + 重试（full jitter）
  llm.py       LLM 客户端（Anthropic 兼容），token 近似估计
  parser.py    think / text / tool_use 解析 + 文本兜底
  context.py   系统提示词拼装 + 上下文压缩
  session.py   窗口状态与落盘（transcript.jsonl / summary / todo）
  store.py     会话目录管理
  memory.py    三层记忆：工作 / 情景 / 语义
  trace.py     运行痕迹（JSONL + ANSI）
  loop.py      第 1~4 步循环：Agent.run_turn
  tools/       工具注册、schema 推导、分发、超时
  __main__.py  命令行入口
```

### 四步循环（loop.py:113）

1. **收到输入** → 追加到 session（落盘）
2. **决定** → 一次 LLM 调用；解析出 `tool_use` 则"行动"，纯文本则"回答"
3. **行动** → 并行执行工具（`ThreadPoolExecutor`），`{ok,result/error}` 回填
4. **继续/结束** → 有 `tool_use` 则回到第 2 步，无则返回最终答案

护栏：`MAX_TURNS`（默认 12）轮内必须收敛；同一签名重复 `MAX_REPEAT_CALL`（默认 3）次即告警收尾；`max_tokens` 截断自动续写；空回复提示重答。

### 记忆：何时读取、放在哪里

> 记忆分三层：工作记忆=当前回合消息；情景记忆=窗口会话记录+滚动摘要；语义记忆=跨窗口长期事实（`.memory/<user>/facts.json`）。

- **读取时机**：每个用户回合开始、模型发言之前（loop.py 中 `memory.recall` 先行）。
- **放置位置**：召回结果注入**系统提示词**的 `Long-term memory` 段落，刻意**不**混入 user 回合——记忆是上下文，不是新指令。
- **写入时机**：召回是读路径，不落盘（只在内存里累加命中计数，关闭窗口时统一 flush）；只有 `remember()` 立即写文件。写入按 id 与磁盘现状合并，两个窗口先后存事实不会互相抹掉。
- **写入时机**：模型主动调用 `remember` 工具（即时）；窗口结束时对整段会话做一次 `extract_facts` 提取。
- **写入门槛**：过短 / 疑似指令 / 疑似提问 / 含时间词 / 数值型工具输出 → 拒绝；归一化去重；TTL 90 天。
- **召回方式**：关键词打分（BM25 风格，零嵌入、零额外 LLM 调用），命中数累计。

### 上下文压缩

超出 `CONTEXT_BUDGET`（默认 24000 估计 token）时：保留首条 user 锚点 + 最近 `KEEP_RECENT_MESSAGES` 条原文，中间段交摘要模型；`tool_use`/`tool_result` 必须成对移动；摘要**折叠进锚点 user 消息**以保持 user/assistant 交替；摘要失败退化为机械淘汰最早完整回合，**永不撕裂工具对**。

## 工具（6 个）

| 工具 | 说明 |
|---|---|
| `calculator` | AST 白名单四则运算（无 eval） |
| `search` | 模拟搜索（本地语料 BM25 词法打分） |
| `weather` | 天气查询。默认 mock（确定性假数据）；`WEATHER_BACKEND=wttr.in` 走真实 wttr.in（免 key），不可达时自动回退 mock |
| `todo` | 窗口内待办 add/list/done/remove/clear |
| `read_docs` | 读取 `docs/` 项目文档（list/search/read/section） |
| `remember` | 把一条事实写入 `.memory/<user>/facts.json`，经写入门控与去重。窗口关闭时的记忆抽取也走同一后端 |

## 数据分布

- `.sessions/<user>/<id>/` 每个窗口一目录（对话、摘要、待办）
- `.traces/` 每次运行全量痕迹（JSONL 整块快照）
- `.memory/<user>/facts.json` 长期记忆
- 均被 `.gitignore` 排除

## 输出与流式

- REPL 下 LLM 回复**流式输出**：`agent` 一行随 token 增量增长（thinking 与工具调用整块展示）
- `.traces/*.jsonl` 是不变的事件快照（每次含完整文本），不受流式影响
- `--demo` 模式不渲染过程流，只打印 `you>/ai>` 问答

## 目录说明

- `docs/agent-handbook.md` 使用手册、`docs/architecture.md` 系统设计
- `PROMPT_LOG.md` AI 提示词与问题解决过程记录（含关键探针）
- `tests/` 144 项离线 + 6 项 live