# SILP — Semantic Interlingua Layer Protocol

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21396849-blue)](https://doi.org/10.5281/zenodo.21396849)
[![IETF Draft](https://img.shields.io/badge/IETF-draft--hwang--silp--protocol--00-blue)](https://datatracker.ietf.org/doc/draft-hwang-silp-protocol/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *A black-box, text-interface payload codec for cross-model agent communication. SILP does not access or manipulate model-internal latent representations. It is a protocol layer, not a prompt compression tool — designed for honest, auditable agent-to-agent communication.*

**Paper**: [Zenodo](https://doi.org/10.5281/zenodo.21396849) · **IETF Draft**: [datatracker](https://datatracker.ietf.org/doc/draft-hwang-silp-protocol/) · **Code**: This repository · **Data**: All raw experimental data included

## What is SILP?

SILP (Semantic Interlingua Layer Protocol) provides a candidate semantic payload layer for MCP/A2A messages, where payload encoding is not yet standardized. It compiles a coarse-grained action-slot intermediate representation (IR) into multiple pluggable surface frontends — code-like function-call syntax, pure JSON, natural language, and ML-compressed text — each designed to exploit the shared training priors of contemporary LLMs.

### Key Finding

The central hypothesis predicted that **shared syntactic priors** (code-format familiarity) contribute more to cross-model comprehension than **shared vocabulary priors** (word-level semantics). Our ablation results **contradict** this prediction: removing semantic vocabulary caused a **2.8× larger performance drop** than shuffling syntactic order (85.2 vs. 30.8 percentage points), though both perturbations produced statistically significant effects.

## Quick Start

```powershell
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install CPU-only PyTorch (must be installed separately)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the project (development mode + all dependencies)
pip install -e ".[ml,api,analysis,dev]"

# 4. Run tests
pytest

# 5. Try the CLI
silpc frontends
silpc validate examples/case1.json
silpc compile examples/case1.json -f code
```

## Project Structure

```
polyglotir/
├── src/silp/
│   ├── ir/            # Layer 1: Semantic IR (Schema + Validator)
│   ├── frontend/      # Layer 2: Pluggable frontends (code/JSON/natural/...)
│   ├── negotiation/   # Layer 3: Meta-protocol (handshake/session/errors)
│   └── bench/         # Layers 4-5: Optimization + Migration screening
├── scripts/           # Experiment scripts (tokenizer census, benchmarks, etc.)
├── tests/             # Test suite
├── data/              # Experimental data (raw + processed + figures)
├── latex/             # LaTeX source for the paper
├── ietf-draft/        # IETF Internet-Draft source (XML + text + HTML)
└── examples/          # Example IR files
```

## Five-Layer Protocol Stack

| Layer | Module | Description |
|-------|--------|-------------|
| 1. Application | `silp.ir` | Semantic IR (JSON-serialized action-slot structure) |
| 2. Surface | `silp.frontend` | Pluggable frontends: code, JSON, natural, nl_json, llmlingua2 |
| 3. Meta-protocol | `silp.negotiation` | Dynamic frontend negotiation, session management, error codes |
| 4. Optimization | `silp.bench` | Multi-objective fitness function + genetic algorithm evolution |
| 5. Migration | `silp.bench` | Small-model → large-model ranking preservation screening |

## Experimental Phases

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 0 | Weeks 1–2 | Cross-tokenizer census + task design (27 cases × 9 categories) |
| 0.5 | Week 3 | Smoke test + compile.lock physical freezing → go/no-go gate |
| 1 | Weeks 4–7 | Primitive whitelist + Validator + MVP round-trip |
| 2 | Weeks 8–12 | **Generalizability benchmark matrix** (5 frontends × 5 models × 27 cases = 675 runs) |
| 3 | Weeks 13–18 | Ablation + entropy analysis + heartbeat (2,160 multi-turn runs) |
| 4 | Future | Automatic evolution + negotiation + cross-language |

## Results Summary

- **Code frontend**: 85.9% average pass rate (highest), statistically tied with JSON (84.4%) and natural language (83.0%)
- **Ablation**: Skeleton (vocabulary removed) collapses to 1.2%; Shuffled (order disrupted) retains 55.6%
- **Compression**: LLMLingua-2 (rate=0.5) achieves 45.8% compression but only 25.9% pass rate
- **Multi-turn**: No statistically significant error propagation across 15 turns (Fisher exact test, p ≥ 0.10)
- **Total**: 3,159 model invocations across 4 phases

## Design Principles

- **Not a compression tool** — SILP is a protocol layer; compression is a side effect, not the goal
- **No jailbreak content** — Only carries legitimate task instructions
- **All encodings are losslessly decodable** via `silpc` — no untraceable steganography
- **No model-internal access** — Pure text interface, black-box

## IETF Internet-Draft

This project's protocol specification has been submitted as an IETF
Internet-Draft (Informational, Independent Submission):

- **Draft**: [draft-hwang-silp-protocol-00](https://datatracker.ietf.org/doc/draft-hwang-silp-protocol/)
- **Status**: Active (submitted 2026-07-18)
- **Source files**: see [`ietf-draft/`](./ietf-draft/) folder
- **Community discussion**: introduced on the IETF `agent2agent` mailing list, which discusses standardization of AI agent communication protocols -- [see thread](https://mailarchive.ietf.org/arch/msg/agent2agent/qRDDvKJ4Cmu64xawdoQ9y645tSc/)

The draft formally specifies the SILP protocol described in this repository.

## Citation

If you use SILP in your research, please cite:

```bibtex
@misc{https://doi.org/10.5281/zenodo.21396849,
  doi = {10.5281/ZENODO.21396849},
  url = {https://doi.org/10.5281/zenodo.21396849},
  author = {Hwang, Juwan},
  keywords = {LLM, agent communication, protocol, interlingua, prompt compression},
  language = {en},
  title = {SILP : A SEMANTIC INTERLINGUA LAYER PROTOCOL FOR CROSS-MODEL AGENT COMMUNICATION},
  publisher = {Zenodo},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
