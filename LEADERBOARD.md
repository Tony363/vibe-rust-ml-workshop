# Leaderboard

CI scores every PR using deterministic scorers with fixed RNG seed (1) for fair comparison.

## Iris (Easy)

| Rank | Participant | Accuracy | Model | PR |
|------|-------------|----------|-------|----|
| -    | Baseline    | 93.3%    | DecisionTree (Entropy, depth=10) | - |

## Wine Quality (Hard)

| Rank | Participant | Accuracy | Model | PR |
|------|-------------|----------|-------|----|
| -    | Baseline    | 53.9%    | DecisionTree (Gini, depth=5) | - |

## How to Submit

1. Create a branch from `submissions`
2. Edit `src/lib.rs` -- change the Iris and/or Wine functions
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
