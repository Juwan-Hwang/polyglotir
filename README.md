# SILP — Semantic Interlingua Layer Protocol

> *A black-box text-interface payload codec; it does NOT access or manipulate
> model-internal latent representations. It is a protocol layer for
> model-to-model communication, not a prompt compression tool — designed for
> honest, auditable agent-to-agent communication.*

SILP provides a candidate semantic payload layer for MCP/A2A messages.
The core hypothesis under test: **shared syntax prior > shared vocabulary prior**
— code/structured-grammar encoding transmits intent across models more reliably
than natural-language vocabulary.

## 快速开始 (Windows)

```powershell
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装 CPU 版 torch（必须先装，单独装）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目（开发模式 + 全部依赖）
pip install -e ".[ml,api,analysis,dev]"

# 4. 运行测试
pytest

# 5. 试用 CLI
silpc frontends
silpc validate examples/case1.json
silpc compile examples/case1.json -f code
```

## 项目结构

```
polyglotir/
├── src/silp/
│   ├── ir/            # 第1层：语义 IR（Schema + Validator）
│   ├── frontend/      # 第2层：插拔式多前端（代码/数学/知识引用/自然语言）
│   ├── negotiation/   # 第3层：元协议（握手/会话/错误码）
│   ├── bench/         # 第4-5层：优化层 + 迁移筛选层
│   └── cli/           # silpc 命令行工具
├── scripts/           # 实验脚本（分词器普查、smoke test 等）
├── tests/             # 测试套件
├── data/              # 论文数据（详见 data/README.md）
└── docs/              # 文档
```

## 五层协议栈

| 层 | 模块 | 作用 |
|---|---|---|
| 1 应用层 | `silp.ir` | 语义 IR（JSON Schema 任务槽位） |
| 2 表面层 | `silp.frontend` | 插拔式多前端（代码默认） |
| 3 元协议层 | `silp.negotiation` | 动态握手、状态管理、错误回退 |
| 4 优化层 | `silp.bench` | 适应度函数 + GA 自动进化 |
| 5 迁移筛选层 | `silp.bench` | 小模型→闭源模型排序保持 |

## 实施路线（18 周）

| 阶段 | 周期 | 核心交付 |
|---|---|---|
| 0 | 1–2 周 | 本地小模型 + 跨分词器普查 + 任务集 |
| 0.5 | 1 周 | Smoke Test + compile.lock 冻结 → 判定门 |
| 1 | 3–4 周 | 原语白名单 + Validator + MVP |
| 2 | 5–8 周 | **通用性基准矩阵**（前端×模型，含闭源） |
| 3 | 9–12 周 | 权衡曲线 + 剥离实验 + 人类可读性 |
| 4 | 13–18 周 | 自动进化 + 协商 + 跨语言 |

## 红线

- **不是压缩工具** — 与 LLMLingua 区别在协议层非压缩层
- **绝不沾越狱** — 仅承载正常任务
- **所有编码可经 `silpc` 无损解码** — 无不可追溯隐写

## 许可证

MIT
