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

一个窗口请只开一个进程——这不是建议，是强行执行的：`open` 一个已存在的窗口会拿 `<session>/.lock` 排它锁，第二个进程打开同一个窗口会直接报错退出（第二个进程拿锁失败时，`min-agent` 会在 stderr 打印是哪个 pid 占着并返回退出码 2）。同一用户的不同窗口互不干扰，可以各开各的进程。锁由操作系统持有，进程被杀也不会留下"锁死"的残留。

## 礼貌清单

- 输入 `/help` 查看窗口内命令，`/quit` 离开。
- 离开窗口时会做一次"记忆入库"：把本窗口里值得长期记住的事实写入 `.memory/<user>/facts.json`。

## 工具

模型按需调用；工具失败返回 `{ok: false, error}` 结构——不会让进程崩溃，而是把错误信息交还给模型自己修正或向用户说明。

- `calculator` 四则运算（安全 AST 解析）
- `search` 模拟网络搜索（本地小型语料 BM25 词法打分）
- `weather` 天气查询。默认 `mock`（确定性假数据，无需联网）；在 `.env` 里设 `WEATHER_BACKEND=wttr.in` 即可切到真实 wttr.in 预报（免费、无 key），服务不可达时自动回退 mock
- `todo` 本窗口的待办清单，支持 add/list/done/remove/clear
- `read_docs` 读取 `docs/` 下的项目文档

## 数据存放（仓库根目录下）

- `.sessions/<user>/<session-id>/` 每个窗口一个目录：对话记录、摘要、待办
- `.traces/` 每次运行的工具调用、思考过程等 JSONL 痕迹
- `.memory/<user>/facts.json` 跨窗口的长期记忆
- 记忆的写入策略：**读路径不落盘**——`recall()` 只在内存里累加命中计数，关闭窗口时
  统一 flush；`remember()` 立即落盘。写入是"读-改-写"，按 id 与磁盘上其他窗口刚写入的
  条目合并，所以两个窗口先后存事实不会互相抹掉。文件损坏时降级为"无记忆"而不是报错退出。
- 以上目录都在 `.gitignore` 中，不会被提交