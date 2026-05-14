# ItaliaBench

> Open benchmark for measuring LLM hallucinations on Italian knowledge.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC-BY 4.0](https://img.shields.io/badge/Dataset-CC--BY%204.0-blue.svg)](data/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

🇮🇹 [Versione italiana](README.it.md)

ItaliaBench measures how well large language models know facts that matter in
Italy: civil law, tax rules, geography, history, and public administration
procedures. It is designed to be **reproducible**, **resistant to gaming**,
and **easy to extend** through community contributions.

> ⚠️ **Pre-release.** v0.1.0 is under active development. The first stable
> dataset will include 100 verified questions across 5 categories.

## Why

General-purpose LLMs are trained on a mostly English corpus and are known to
hallucinate when answering domain-specific questions about Italy: wrong VAT
rates, invented Civil Code articles, confused PA procedures. Existing
benchmarks (MMLU, TruthfulQA) under-sample Italian content and rarely include
items that depend on current Italian regulations.

ItaliaBench provides:

- A curated set of fact-based questions with verified sources.
- A hybrid scoring engine: deterministic keyword/regex checks plus opt-in
  LLM-as-judge for ambiguous cases.
- Adapters for the major providers (OpenAI, Anthropic, Google, Ollama) so
  comparisons are reproducible at `temperature=0`.
- Versioned, hashed datasets with a small set of secret canary questions to
  detect contamination over time.

## Quick start

```bash
pip install italiabench[openai,anthropic]

italiabench run --model gpt-4o --category fisco
italiabench run --model claude-opus-4-7
italiabench compare --models gpt-4o,claude-opus-4-7 --report html
```

## Categories (v0.1.0)

| Category           | Examples                                             |
|--------------------|------------------------------------------------------|
| `diritto`          | Civil Code articles, GDPR-IT, consumer code          |
| `fisco`            | VAT rates, tax regimes, deadlines                    |
| `geografia_pa`     | Municipalities, ZIP codes, ASL territorial units     |
| `storia_cultura`   | Historical events, governments, cultural references  |
| `procedure`        | SPID, PEC, electronic invoicing, public procedures   |

## Roadmap

- **v0.1.0** — 100 questions (20 × 5 categories), 4 provider adapters, CLI.
- **v0.2.0** — community-driven scaling to 500 questions.
- **v1.0.0** — public leaderboard, signed dataset releases.
- **v1.1.0** — regional dialects subset.
- **v1.2.0** — `aderenza_normativa` category (regulatory compliance scoring).

## Contributing

The dataset grows through contributions. To propose new questions, see
[CONTRIBUTING.md](CONTRIBUTING.md). Every question must include an authoritative
source and a `last_verified` date.

## Citation

```bibtex
@misc{italiabench2026,
  title  = {ItaliaBench: an open benchmark for Italian knowledge in LLMs},
  author = {{ItaliaBench contributors}},
  year   = {2026},
  url    = {https://github.com/jonni2712/italiabench}
}
```

## License

- **Code**: [MIT](LICENSE)
- **Dataset**: [CC-BY 4.0](data/LICENSE)
