# Leaderboard

CI scores every PR using deterministic scorers with fixed RNG seed (1) for fair comparison.

*Updated automatically when submissions are merged.*

## Iris (Easy)

| Rank | Participant | Accuracy | Model | PR |
|------|-------------|----------|-------|----|
| 🥇 | @AnisahLM | 96.7% | DecisionTree (Gini, depth=8) | #16 |
| 🥈 | @Tony363 | 96.7% | DecisionTree (Gini, unlimited depth) | #15 |
| 🥉 | @sarahbouayad | 96.7% | Medina Creative Model (Gini, no depth limit) | #14 |
| 4 | @Tony363 | 96.7% | DecisionTree (Gini, depth=5) | #4 |
| 5 | @Tony363 | 93.3% | DecisionTree (Entropy, depth=3) | #10 |
| -    | Baseline    | 93.3%    | DecisionTree (Entropy, depth=10) | - |

## Wine Quality (Hard)

| Rank | Participant | Accuracy | Model | PR |
|------|-------------|----------|-------|----|
| 🥇 | @AnisahLM | 61.8% | DecisionTree (Entropy, depth=20, normalised) | #16 |
| 🥈 | @sarahbouayad | 54.9% | DecisionTree (Gini, depth=7) | #14 |
| 🥉 | @Tony363 | 54.5% | DecisionTree (Gini, depth=7) | #15 |
| -    | Baseline    | 53.9%    | DecisionTree (Gini, depth=5) | - |

## How to Submit

1. Create a branch from `submissions`
2. Edit `src/lib.rs` -- change the Iris and/or Wine model functions
3. Test locally:
   - `cargo run --bin score --release` (Iris)
   - `cargo run --bin score_wine --release` (Wine Quality)
4. Push your branch and open a PR targeting the `submissions` branch
5. CI will automatically score both challenges and post a combined leaderboard

## Rules

- Only modify `src/lib.rs` and `Cargo.toml` (to add dependencies)
- Must use linfa algorithms
- The scorers use their own seeded RNG and train/test split -- you cannot influence them
- Highest accuracy wins
