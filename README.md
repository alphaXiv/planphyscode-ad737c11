# PlanPhys released-checkpoint reproduction — blocked at inference

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/planphyscode-ad737c11/blob/main/notebooks/planphys_reproduction.py)

This repository tests two pretraining claims from [arXiv:2607.24720](https://arxiv.org/abs/2607.24720): whether explicit state-transition reasoning beats direct action prediction, and whether limited long-horizon exposure improves composition beyond short-only training. We reconstructed the unreleased gym from the public Graph A records and released schemas, selected 160 tasks per domain and difficulty, and prepared matched checkpoint-5400 and checkpoint-9000 evaluations.

**Assessment: behavioral reproduction blocked.** The paper reports large gains, but our observed behavioral numbers are unavailable: all four model-evaluation branches reached a deterministic tokenizer/model compatibility error after their two allowed setup attempts. The independently reconstructed verifier did run successfully on Kubernetes and matched every released gold record, but that method check is not a proxy for model behavior.

| Target | Paper number | Observed number | Assessment |
|---|---:|---:|---|
| World model, middle-horizon avg@8 | 46.6% → 85.9% | Not measured | Blocked before inference |
| World model, long-horizon avg@8 | 22.7% → 68.5% | Not measured | Blocked before inference |
| Short-only → 5%-long, middle pass@8 | 0.83% → 47.29% | Not measured | Blocked before inference |
| Short-only → 5%-long, long pass@8 | 0.00% → 11.46% | Not measured | Blocked before inference |
| Reconstructed Graph A verifier | Gold records imply 100% | 100% gold success, 100% step-count agreement, 100% corrupted-terminal rejection (n=1,440 each) | Aligned method audit |

Scope and substitutions: released 100M checkpoints only; Graph A public test pools only; deterministic exact-step stratification with seed `260724720`; reconstructed inventory transitions because the gym is still unreleased. Kubernetes was used on a cluster with NVIDIA RTX PRO 6000 Blackwell GPUs. Peak concurrent GPU allocation was 16; measured end-to-end Kubernetes wall time was 337 seconds (0.094 hours), from the first job creation to the successful audit completion.

Read the [illustrated report](reports/planphys/report.md) or the [self-contained marimo walkthrough](notebooks/planphys_reproduction.py).

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and repaired reference implementation | Not run as an experiment (publication surface) | Presentation-only | — |
| [World model final](https://github.com/alphaXiv/planphyscode-ad737c11/tree/orx/world-model-final-checkpoint) | CoT world model vs direct action, checkpoint 9000 | `bash scripts/run_reproduction.sh` | Two setup failures; verifier audit passed before tokenizer incompatibility | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [World model intermediate](https://github.com/alphaXiv/planphyscode-ad737c11/tree/orx/world-model-intermediate-checkpoint) | Same comparison, checkpoint 5400 | `bash scripts/run_reproduction.sh` | Two setup failures; no behavioral metric | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Long-horizon exposure final](https://github.com/alphaXiv/planphyscode-ad737c11/tree/orx/long-horizon-exposure-final-checkpoint) | Short-only vs S100/M100/L5, checkpoint 9000 | `bash scripts/run_reproduction.sh` | Two setup failures; no behavioral metric | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Long-horizon exposure intermediate](https://github.com/alphaXiv/planphyscode-ad737c11/tree/orx/long-horizon-exposure-intermediate-checkpoint) | Same comparison, checkpoint 5400 | `bash scripts/run_reproduction.sh` | Two setup failures; no behavioral metric | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Graph A verifier audit](https://github.com/alphaXiv/planphyscode-ad737c11/tree/orx/graph-a-verifier-audit) | Gold, step-count, and terminal-corruption checks | `bash scripts/run_reproduction.sh` | Completed: all three checks 100% on 1,440 tasks | Kubernetes CPU job; RTX PRO 6000 Blackwell cluster |

## Reproduction implementation

The evaluator selectively downloads only two configured checkpoints plus nine Graph A test files and the matching item/schema maps. It builds the current inventory, maps concrete names to abstract recipe nodes, applies one- or two-input transitions without consuming prerequisites, and declares success only when the concrete target enters inventory. Model evaluation uses temperature 0.4, eight trajectories per task, a 20-turn cap, strict and lenient parsing, and paired task bootstrap intervals.

```bash
bash scripts/run_reproduction.sh
```

The command is intended for the committed Kubernetes manifest. Expensive inference was not rerun after the repair cap; the `main` implementation includes the diagnosed `token_type_ids` compatibility fix for future work.

## Upstream release

This repository began from the public [PlanPhysCode release](https://github.com/Quester-one/PlanPhysCode). Checkpoints and datasets remain on their Apache-2.0 Hugging Face repositories and are downloaded at runtime rather than redistributed here.
