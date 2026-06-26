# 数据目录结构 — 论文数据管理策略

> **核心原则：可复现性优先。** 任何人拿到这份代码+数据，应能完全复现论文中的每一张表、每一张图。

## 目录结构

```
data/
├── raw/            # 原始实验输出（不可修改，immutable）
│   ├── phase0.5/   #   按阶段分子目录
│   │   ├── smollm-360m/       # 按模型分子目录
│   │   │   ├── case_001_code.jsonl
│   │   │   ├── case_001_natural.jsonl
│   │   │   └── ...
│   │   ├── qwen2.5-0.5b/
│   │   ├── gpt-4o/
│   │   └── ...
│   ├── phase2/
│   └── phase3/
│
├── processed/      # 清洗 + 指标计算后的结构化数据（CSV / Parquet）
│   ├── phase0.5/
│   │   ├── success_rates.csv        # 各模型×前端的首过通过率
│   │   ├── tokenizer_census.csv     # 跨分词器 token 数普查
│   │   └── compile.lock             # silpc 生成的编译冻结日志（副本）
│   ├── phase2/
│   └── phase3/
│
├── figures/        # 论文最终图片（PDF / SVG / PNG，300 DPI）
│   ├── fig1_success_matrix.pdf
│   ├── fig2_tradeoff_curve.pdf
│   └── ...
│
├── tables/         # 论文最终表格（LaTeX / CSV）
│   ├── tab1_benchmark_matrix.tex
│   └── ...
│
└── metadata/       # 实验配置与运行日志
    ├── model_configs.json   # 每个模型的精确配置（版本、参数、温度等）
    ├── run_logs/            # 每次实验运行的完整日志
    └── task_set/            # 实验用任务集（固定模板，禁人工润色）
```

## 各层职责

### `data/raw/` — 原始输出（写一次，永不改）

- **内容**：模型每次返回的原始文本、解码结果、时间戳、模型名称、前端类型
- **格式**：JSONL（每行一条记录，方便流式处理）
- **纪律**：一旦写入，绝不修改。如果实验出错，重新跑一遍写到新文件
- **Git 策略**：Phase 0.5 的小规模数据（~240 条）直接入 git；大规模数据（Phase 2+）用 git-LFS 或外部存储，git 中保留校验和

### `data/processed/` — 结构化分析数据

- **内容**：从 raw 计算出的指标 CSV（通过率、token 数、压缩率、PPL 等）
- **生成方式**：`scripts/` 下的分析脚本从 `raw/` 生成，可随时重跑
- **Git 策略**：全部入 git（CSV 文件小，且是论文表格的直接来源）

### `data/figures/` — 论文图片

- **内容**：matplotlib/seaborn 生成的最终论文图
- **生成方式**：`scripts/make_figures.py` 从 `processed/` 数据生成
- **Git 策略**：入 git（PDF/SVG 文件不大）

### `data/tables/` — 论文表格

- **内容**：LaTeX 格式的论文表格
- **Git 策略**：入 git

### `data/metadata/` — 实验配置

- **内容**：模型版本、API 参数、随机种子、任务集定义
- **纪律**：每次实验运行前冻结一份配置快照，确保可复现

## 数据流

```
  固定任务集 (metadata/task_set/)
        │
        ▼
  模型推理 → data/raw/{phase}/{model}/case_NNN_{frontend}.jsonl
        │
        ▼
  分析脚本 (scripts/) → data/processed/{phase}/*.csv
        │
        ▼
  制图脚本 (scripts/) → data/figures/*.pdf + data/tables/*.tex
        │
        ▼
  论文 (paper/)
```

## compile.lock 的特殊地位

`.silp/compile.lock` 是 silpc 自动生成的编译冻结日志：
- 记录 IR 哈希 → {前端输出, 时间戳, git commit}
- **必须入 git**（这是"首次原始通过率"的不可篡改证据）
- 重新编译必须先 `rm .silp/compile.lock`，git 会记录这次删除
- 论文附录应附上 compile.lock 内容作为可审计证据

## 命名约定

```
raw/{phase}/{model}/case_{NNN}_{frontend}.jsonl
```

- `phase`: `phase0.5`, `phase2`, `phase3`, `phase4`
- `model`: `smollm-360m`, `qwen2.5-0.5b`, `tinyllama-1.1b`, `gpt-4o`, `claude-3.5`, `gemini-pro`
- `NNN`: 三位数序号，零填充（`001`, `002`, ...）
- `frontend`: `code`, `natural`, `math`, `knowledge`, `json`

每条 JSONL 记录格式：

```json
{
  "case_id": "001",
  "frontend": "code",
  "model": "smollm-360m",
  "ir_hash": "a1b2c3d4e5f6a7b8",
  "encoded": "if loc(me,Beijing,t+1am): cancel(flight,t+1pm); email(zhangsan)",
  "model_response": "<模型的完整回复>",
  "judge_result": "pass",
  "judge_reason": "<LLM-as-judge 的判定理由>",
  "timestamp": "2025-01-15T10:30:00Z",
  "first_pass": true
}
```
