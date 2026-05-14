# ItaliaBench

> Benchmark open per misurare le allucinazioni degli LLM sulla conoscenza italiana.

🇬🇧 [English version](README.md)

ItaliaBench misura quanto bene i grandi modelli linguistici conoscono i fatti
che contano in Italia: diritto civile, normativa fiscale, geografia, storia e
procedure della pubblica amministrazione. È progettato per essere
**riproducibile**, **resistente al gaming** e **facile da estendere** tramite
contributi della community.

> ⚠️ **Pre-release.** la v0.1.0 è in sviluppo attivo. Il primo dataset stabile
> includerà 100 domande verificate su 5 categorie.

## Perché

Gli LLM general-purpose sono addestrati su un corpus prevalentemente inglese
e sono noti per allucinare quando rispondono a domande specifiche sull'Italia:
aliquote IVA sbagliate, articoli del Codice Civile inventati, procedure PA
confuse. I benchmark esistenti (MMLU, TruthfulQA) sotto-campionano i contenuti
italiani e raramente includono item che dipendono dalla normativa italiana
corrente.

ItaliaBench fornisce:

- Un set curato di domande fattuali con fonti verificate.
- Un engine di scoring ibrido: controlli deterministici via keyword/regex più
  LLM-as-judge opt-in per i casi ambigui.
- Adapter per i principali provider (OpenAI, Anthropic, Google, Ollama) per
  confronti riproducibili a `temperature=0`.
- Dataset versionati e hashed con un piccolo set di canary question segrete
  per detectare contamination nel tempo.

## Quick start

```bash
pip install italiabench[openai,anthropic]

italiabench run --model gpt-4o --category fisco
italiabench run --model claude-opus-4-7
italiabench compare --models gpt-4o,claude-opus-4-7 --report html
```

## Categorie (v0.1.0)

| Categoria          | Esempi                                                  |
|--------------------|---------------------------------------------------------|
| `diritto`          | Articoli del Codice Civile, GDPR-IT, codice consumo     |
| `fisco`            | Aliquote IVA, regimi fiscali, scadenze                  |
| `geografia_pa`     | Comuni, CAP, ASL territoriali                           |
| `storia_cultura`   | Eventi storici, governi, riferimenti culturali          |
| `procedure`        | SPID, PEC, fatturazione elettronica, procedure PA       |

## Roadmap

- **v0.1.0** — 100 domande (20 × 5 categorie), 4 adapter provider, CLI.
- **v0.2.0** — scaling a 500 domande tramite community.
- **v1.0.0** — leaderboard pubblica, release del dataset firmate.
- **v1.1.0** — subset dialetti regionali.
- **v1.2.0** — categoria `aderenza_normativa` (scoring conformità).

## Contribuire

Il dataset cresce con i contributi. Per proporre nuove domande, vedi
[CONTRIBUTING.md](CONTRIBUTING.md). Ogni domanda deve includere una fonte
ufficiale e una data `last_verified`.

## Citation

```bibtex
@misc{italiabench2026,
  title  = {ItaliaBench: an open benchmark for Italian knowledge in LLMs},
  author = {{ItaliaBench contributors}},
  year   = {2026},
  url    = {https://github.com/jonni2712/italiabench}
}
```

## Licenza

- **Codice**: [MIT](LICENSE)
- **Dataset**: [CC-BY 4.0](data/LICENSE)
