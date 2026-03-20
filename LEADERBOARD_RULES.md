# ML Leaderboard Rules

## Goal

Achieve the highest accuracy on the Iris test set.

## How to Submit

1. Create a branch from `submissions`
2. Edit `src/lib.rs` -- modify `build_and_predict()` and `model_name()`
3. Open a PR
4. A GitHub Action will score your model and post a ranked leaderboard

## What to Change

Only modify these two functions in `src/lib.rs`:

- **`build_and_predict(train, test_features)`** -- Train your model and return predictions
- **`model_name()`** -- Name your model (shown on the leaderboard)

## Allowed

- Any `linfa-*` crate (trees, SVM, KNN, linear, logistic, etc.)
- Any algorithm, any hyperparameters
- Feature engineering using `ndarray` operations
- Adding new `linfa-*` dependencies to `Cargo.toml`

## Not Allowed

- Modifying the scorer (`src/bin/score.rs`)
- Hardcoding predictions
- Using non-linfa ML crates
- Changing the random seed or split ratio

## Scoring

- Fixed random seed: `1`
- Train/test split: 80/20
- Metric: accuracy = correct predictions / total test samples
- Deterministic -- same code always gets the same score

## What You Get on Your PR

CI posts a detailed performance report as a comment on your PR:

- **Accuracy comparison chart** -- Mermaid bar chart showing your score vs other submissions and the baseline
- **Ranked leaderboard** -- standings across all open submissions
- **Confusion matrix** -- shows which classes your model confuses
- **Per-class metrics** -- precision, recall, and F1 score for Setosa, Versicolor, and Virginica

When your PR is merged, the unified [LEADERBOARD.md](LEADERBOARD.md) on the `submissions` branch is updated automatically.

## Quick-Start Vibe Prompts

Try giving these prompts to your AI coding assistant:

> "Change the model in src/lib.rs to use KNN with k=5 instead of a decision tree. Use linfa-nn."

> "Switch to a random forest approach -- train multiple decision trees with different depths and take a majority vote."

> "Use linfa-logistic to train a multinomial logistic regression model instead of a decision tree."

> "Try linfa-svm with an RBF kernel for the Iris classifier."

> "Reduce max_depth to 3 and switch from Entropy to Gini to see how accuracy changes."
