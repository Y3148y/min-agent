# min-agent handbook

min-agent 是一个从零开始实现的最小 Agent，只有 **四步循环**：收到输入 -> 决定回答或用工具 -> 调用工具 -> 决定继续或结束。整个报告不依赖任何现成的 agent 框架。

## 运行方式

```bash
pip install -e .
cp .env.example .env   # 填入 ANTHROPIC_API_KEY

min-agent --user alice --session 周末规划   # 交互式聊天窗口
min-agent --demo                          # 内置演示
min-agent --user alice sessions           # 列出 alice 的所有窗口
min-agent --user alice trace --session 周末规划
min-agent --user alice tools              # 列出模型可用的工具
```

一个用户的多个窗口（`--session`）相互独立：各自的对话记录、待办清单、运行痕迹都分开存放。两个进程可以同时操作不同窗口。

## 礼貌清单

- 输入 `/help` 查看窗口内命令，`/quit` 离开。
- 离开窗口时会做一次"记忆入库"：把本窗口里值得长期记住的事实写入 `.memory/<user>/facts.json`。

## 工具

模型按需调用；工具失败返回 `{ok: false, error}` 结构——不会让进程崩溃，而是把错误信息交还给模型自己修正或向用户说明。

- `calculator` 四则运算（安全 AST 解析）
- `search` 模拟网络搜索（本地小型语料 BM25 词法打分）
- `weather` 模拟天气查询（按城市名做确定性假数据）
- `todo` 本窗口的待办清单，支持 add/list/done/remove/clear
- `read_docs` 读取 `docs/` 下的项目文档

## 数据存放（仓库根目录下）

- `.sessions/<user>/<session-id>/` 每个窗口一个目录：对话记录、摘要、待办
- `.traces/` 每次运行的工具调用、思考过程等 JSONL 痕迹
- `.memory/<user>/facts.json` 跨窗口的长期记忆
- 以上目录都在 `.gitignore` 中，不会被提交