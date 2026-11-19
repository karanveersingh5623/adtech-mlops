# Contributing

This project is built by two contributors who are learning together. The workflow below keeps `main` working, keeps changes reviewable, and makes sure **both people understand every part of the system**.

## Principles

- **Shared ownership.** Nobody owns a component. Both contributors are expected to be able to implement, debug, and explain the data, model, feature, serving, and operations code.
- **Alternate roles.** Split each day into at least two tasks. Alternate who implements and who reviews, so that over the ten days each person builds part of every layer.
- **Small and verifiable.** Prefer several small pull requests that each work and are tested over one large change.
- **Correctness over polish.** If time runs short, drop optional extras before tests or required deliverables.

## Branches

`main` is always runnable, and its tests always pass. Never commit to it directly.

Create one branch per task from an up-to-date `main`:

```bash
git switch main
git pull --ff-only
git switch -c day2/streaming-ctr
```

Naming: `day<N>/<short-topic>` for roadmap work (e.g. `day4/ndcg`, `day6/rank-endpoint`), `docs/<topic>` for documentation, and `fix/<topic>` for bug fixes.

## Commits

- Each commit makes one logical change and leaves the tests passing.
- Write messages in the imperative mood, with a short summary line and an optional body explaining *why*:

  ```text
  Add streaming CTR aggregation by device and category

  Uses running counts so memory is independent of file size.
  Equivalence with pandas groupby is tested for several chunk sizes.
  ```

- Do not commit generated artifacts (`models/`, large data files), virtual environments, or secrets. Run `git diff --staged` before each commit.
- Do not rewrite shared history: no force-pushes to `main` or to branches someone else is reviewing.

## Pull requests

Every pull request description should include:

1. **What** changed and **why**, and the roadmap day or deliverable it addresses.
2. **How it was tested**, with the commands and a short excerpt of their output.
3. **Assumptions, trade-offs, and known limitations.**
4. Any **new dependency**, with a reason and a version constraint.

Keep pull requests focused. A reviewer should be able to read one in about 15 minutes.

## Review checklist

The reviewer checks out the branch and runs it locally, then checks:

- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] The documented command for the new feature works as written.
- [ ] There is no leakage: `clicked`, `converted`, and future events never reach features.
- [ ] New logic has focused tests, including edge cases.
- [ ] Errors are handled explicitly and are not silently swallowed.
- [ ] No secrets, personal data, or large binaries are added.
- [ ] Docs (README status tables, ROADMAP checkboxes, ARCHITECTURE diagrams) are updated where the change affects them.
- [ ] **The reviewer can explain the change in their own words.** If not, ask questions before approving.

Merge only after approval. Prefer **squash merge** for small pull requests so that `main` keeps one clear commit per change.

## Definition of done

A task is complete when:

- The code runs from a command documented in the README or ROADMAP.
- Tests cover the new behavior, where practical.
- Dependencies are recorded with version constraints.
- Results come from actual runs. Metrics and latency numbers are never estimated or copied from elsewhere.
- Assumptions and limitations are written down.
- The other contributor has reviewed it and can reproduce the result.

## Daily routine

1. **Start (10 min):** agree on the day's tasks and acceptance criteria, using [ROADMAP.md](ROADMAP.md), and decide who implements and who reviews each task.
2. **Learn (≈45 min):** study the day's topics. Each person notes one question to discuss.
3. **Build (≈90 min):** implement on feature branches; open pull requests early as drafts.
4. **Validate (≈45 min):** review, run the tests, merge, then run the full suite on `main` together.
5. **Close (5 min):** tick off completed deliverables and write down one trade-off or open question.

## Using AI coding assistants

AI-generated code is treated as an untrusted draft:

- Read every line before committing it.
- Run the tests.
- Verify library behavior against the documentation.
- Be able to explain the code without the assistant.

Each contributor should also complete some implementation and debugging work on their own, without assistance.

## Running tests

```bash
python -m unittest discover -s tests -v
```

Add tests for each new module as `tests/test_<module>.py`. Use small generated fixtures so that the suite stays fast. See the [testing strategy](../README.md#11-testing-strategy).
