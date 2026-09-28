# min-agent 系统设计

## 总体结构

```
min_agent/
  config.py    中心配置：所有可调参数，从 .env 解析一次，测试可覆盖
  errors.py    错误层级 + 重试（full jitter）
  llm.py       大模型客户端（Anthropic 兼容协议），两种估算 token 的启发式
  parser.py    把 LLM 输出拆成 think / text / tool_use；未按格式返回时做文本兜底
  context.py   系统提示词拼装 + 上下文压缩
  session.py   一个窗口的状态与落盘（transcript.jsonl / meta / summary / todo）
  store.py     会话目录的打开与枚举
  memory.py    三层记忆（工作/情景/语义），门控写入 + 关键词召回
  trace.py     运行痕迹：JSONL + ANSI 控制台渲染
  loop.py      第 1~4 步循环：Agent.run_turn
  tools/       工具注册、schema 推导、分发、超时
  __main__.py  命令行入口
```

## 四个关键设计点

### 1. 记忆的时序和位置（报告题目）

- **何时读取**：每个用户回合开始、模型发言之前（`loop.py` 中 `memory.recall` 先行）。
- **放在哪里**：召回结果注入 **系统提示词** 的 `Long-term memory` 段落，而不是混进 user 回合——记忆是上下文而不是命令。
- **何时写入**：模型主动调用 `remember` 工具（即时），以及窗口结束时对整段会话做一次 `extract_facts` 提取。
- **写入门槛**：太短、疑似指令、疑似提问、含时间词、数值型工具输出都会被拒绝；内容做归一化去重。
- **召回方式**：关键词打分（BM25 风格），每条事实会累计命中数；TTL 90 天。

### 2. 上下文压缩防止工具对撕裂

`tool_use` 和它的 `tool_result` 必须成对出现，否则下一次请求会被网关拒绝。压缩时：
- 保留第一条 user 消息作锚点，最近若干条消息原样保留；
- 中间部分交给摘要模型（失败则退化为机械淘汰"最早的完整回合"）；
- 摘要折叠进锚点 user 消息里，从而保持 user/assistant 交替不破坏。

### 3. 工具失败是数据，不是异常

工具调用被包成 `{ok, result/error, latency_ms, truncated}`。进程编码错误（如 `1/0`）用 `is_error` 标记返回给模型，让模型自己判断重试或解释；只有超时、网络等外层错误才走重试后抛错。

### 4. 痕迹、会话与多进程

- `trace.py`：每次工具调用、每次压缩、每条记忆都落 JSONL，供复盘。
- 会话为"每窗口一目录"的纯 JSON 结构，与 SDK 无关；中断后重新加载会自动清理悬空的 `tool_use`/`tool_result` 对。