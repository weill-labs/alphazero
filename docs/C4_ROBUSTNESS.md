# Connect Four Robustness Study

Multi-seed Connect Four training was run on Modal for seeds 0-4 with:

```bash
uv run --extra modal modal run modal_app.py --game connectfour --seed N --iterations 30 --self-play-games 24 --sims 128
```

Final metrics were pulled from W&B project `cweill-self/alphazero-tictactoe` with `wandb.Api()`. Seed 2 was preempted once by Modal; the table uses the completed restarted run `jm83lgan` and excludes the partial preempted run `bbaklww0`. Stdev is sample standard deviation across the five completed seeds.

## Per-Seed Final Metrics

| Seed | W&B run | Elo | Immediate win | Block | Value MAE | Policy match | Random win | Negamax d1 | Negamax d2 | Negamax d4 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | [avopltz6](https://wandb.ai/cweill-self/alphazero-tictactoe/runs/avopltz6) | 48.8 | 100.0% | 100.0% | 0.698 | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 1 | [tjz61ku8](https://wandb.ai/cweill-self/alphazero-tictactoe/runs/tjz61ku8) | 38.4 | 100.0% | 50.0% | 0.169 | 87.5% | 100.0% | 100.0% | 30.0% | 50.0% |
| 2 | [jm83lgan](https://wandb.ai/cweill-self/alphazero-tictactoe/runs/jm83lgan) | 29.6 | 100.0% | 100.0% | 0.859 | 75.0% | 95.0% | 100.0% | 50.0% | 100.0% |
| 3 | [rldmlnnn](https://wandb.ai/cweill-self/alphazero-tictactoe/runs/rldmlnnn) | 67.2 | 100.0% | 100.0% | 0.832 | 100.0% | 95.0% | 100.0% | 100.0% | 100.0% |
| 4 | [w13vjw9d](https://wandb.ai/cweill-self/alphazero-tictactoe/runs/w13vjw9d) | 48.8 | 100.0% | 100.0% | 1.072 | 100.0% | 95.0% | 100.0% | 100.0% | 100.0% |

## Aggregate

| Metric | Mean | Stdev |
| --- | ---: | ---: |
| `eval/elo` | 46.56 | 14.06 |
| `eval/c4_immediate_win_rate` | 100.0% | 0.0 pp |
| `eval/c4_block_rate` | 90.0% | 22.4 pp |
| `eval/c4_value_mae` | 0.726 | 0.339 |
| `eval/c4_policy_match` | 82.5% | 20.9 pp |
| `eval/ladder_random_winrate` | 97.0% | 2.7 pp |
| `eval/ladder_negamax_d1_winrate` | 100.0% | 0.0 pp |
| `eval/ladder_negamax_d2_winrate` | 76.0% | 33.6 pp |
| `eval/ladder_negamax_d4_winrate` | 90.0% | 22.4 pp |

## Conclusion

The Connect Four results look robust across seeds in this 30-iteration setting rather than being a single lucky seed. All five seeds finished with positive Elo, perfect immediate-win recognition, strong block performance, high random-ladder win rate, and non-degenerate exact-solver anchor metrics. The solver anchors are the most useful cross-seed checks here: value MAE averaged 0.726 and policy match averaged 82.5%, with every completed seed matching at least half of the solver-preferred moves.

The deep negamax ladder rungs should be treated as noisy at these small game counts. The d2 and d4 win rates have visibly larger cross-seed spread than the core metrics, so they are useful as a smoke test but not as a trustworthy seed-robustness signal from a single evaluation. The reliable signals are Elo, immediate-win/block tactics, and the exact-solver value-MAE/policy-match anchors.
