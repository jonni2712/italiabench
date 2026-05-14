# Contributing to ItaliaBench

Thanks for considering a contribution. ItaliaBench grows when domain experts
add high-quality, verifiable questions across the five categories.

## Ways to contribute

1. **Add new questions** — the highest-leverage contribution.
2. **Fix or update existing questions** — laws and rates change.
3. **Add provider adapters** — new model APIs.
4. **Improve the scoring engine** — more robust normalization, better LLM
   judging prompts.
5. **Translate documentation** — keeping `README.md` and `README.it.md`
   in sync.

## Quality bar for questions

A question is accepted only if it satisfies all of the following:

1. **Single, unambiguous fact**: avoid open-ended or opinion-based items.
2. **Authoritative source cited**: link or citation to a primary source
   (Gazzetta Ufficiale, Agenzia delle Entrate, Garante Privacy, ISTAT,
   official government portals). Wikipedia is not accepted as primary source
   for fiscal/legal categories.
3. **`last_verified` date** in the form `YYYY-MM-DD`.
4. **Time-stable or clearly time-bounded**: if the answer can change (e.g.
   a tax rate), say "as of YYYY" in the question.
5. **`must_mention` and `must_not_mention` lists**: enough to make scoring
   deterministic without an LLM judge in 80%+ of cases.
6. **Difficulty tagged**: `easy`, `medium`, `hard`.

## Question file format

One question per YAML file under
`data/questions/<category>/<id>.yaml`:

```yaml
id: fisco-iva-001
category: fisco
difficulty: easy
question: |
  Qual è l'aliquota IVA ridotta applicata ai libri in Italia nel 2026?
ground_truth: "4%"
must_mention:
  - "4%"
  - any_of: ["aliquota minima", "super-ridotta"]
must_not_mention:
  - "10%"
  - "22%"
source:
  - "DPR 633/72, Tabella A, parte II, n. 18"
  - "https://www.agenziaentrate.gov.it/..."
last_verified: 2026-04-01
```

## Workflow

1. Open an issue describing the questions you plan to add (avoids duplicate
   work).
2. Fork the repo, create a branch named `questions/<category>-<short-desc>`.
3. Add YAML files. Run `italiabench validate` locally — CI will reject
   malformed files.
4. Open a PR. The PR template asks you to confirm sources and verification
   date.

## Code contributions

```bash
git clone https://github.com/jonni2712/italiabench.git
cd italiabench
pip install -e ".[dev,all]"
pre-commit install
pytest
```

Style: `ruff check .` and `black .` must pass. Type-check with `mypy italiabench`.

## Code of Conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
