# Leaderboard

Scores from the workshop challenge. CI scores every PR using a deterministic scorer (`src/bin/score.rs`) with fixed RNG seed (1) for fair comparison.

*Updated automatically when submissions are merged.*

| Rank | Participant | Accuracy | Model | PR |
|------|-------------|----------|-------|----|
| -    | Baseline    | 93.3%    | DecisionTree (Entropy, depth=10) | - |

## How to Submit

1. Create a branch from `submissions`
2. Edit `src/lib.rs` -- change `build_and_predict()` and `model_name()` to try different algorithms
3. Run `cargo run --bin score --release` locally to check your deterministic score
4. Push your branch and open a PR targeting the `submissions` branch
5. CI will automatically score your code and post a leaderboard comment on the PR

## Rules

- Only modify `src/lib.rs` and `Cargo.toml` (to add dependencies)
- Must use linfa algorithms
- The scorer uses its own seeded RNG and train/test split -- you cannot influence it
- `build_and_predict()` receives training data and test features, returns predictions
- Highest accuracy wins
