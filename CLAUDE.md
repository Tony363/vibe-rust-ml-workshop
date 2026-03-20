# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Workshop demo project: an Iris flower classifier in Rust using the Linfa ML framework. Demonstrates the "vibe coding" workflow in a 45-min workshop. The project includes a library crate with a participant-editable model, a demo binary, and a CI-powered leaderboard system.

Git tags serve as workshop checkpoints (`step-1-scaffold` through `step-4-complete`).

## Build & Run Commands

```bash
cargo build                    # debug build
cargo run --release            # run demo (seed=42, two models compared)
cargo run --bin score --release  # run deterministic scorer (seed=1, lib.rs model only)
cargo clippy                   # lint
cargo fmt                      # format
cargo test                     # run tests (none currently exist)
```

## Architecture

Library crate (`src/lib.rs`) + two binaries (`src/main.rs`, `src/bin/score.rs`):

### src/lib.rs -- Participant model
- `build_and_predict(train, test_features)` -- trains a model and returns predictions
- `model_name()` -- returns model name for display/leaderboard
- This is the only file participants edit for leaderboard submissions

### src/main.rs -- Demo binary
1. **Data loading**: `linfa_datasets::iris()` provides 150 samples with 4 features
2. **Train/test split**: 80/20 shuffle with seed=42
3. **Training**: Runs the participant model from lib.rs + a built-in Entropy tree for comparison
4. **Evaluation**: Accuracy, confusion matrix, sample predictions
5. **Output**: Tables rendered with `comfy-table` using `UTF8_FULL` preset

### src/bin/score.rs -- Deterministic scorer
- Uses fixed seed=1 (different from demo's seed=42) for fair comparison
- Runs only the lib.rs model, outputs JSON with accuracy/model_name
- Used by CI to score leaderboard submissions

### Leaderboard system
- Participants branch from `submissions`, edit `src/lib.rs`, open a PR
- `.github/workflows/leaderboard.yml` scores the PR and posts a ranked leaderboard comment
- Rules in `LEADERBOARD_RULES.md`, standings in `LEADERBOARD.md`

## Key Dependencies

- **linfa** / **linfa-trees** / **linfa-datasets**: ML framework (scikit-learn equivalent for Rust)
- **ndarray**: N-dimensional arrays (used for dataset records)
- **comfy-table**: Terminal table formatting
- **rand**: Dataset shuffling

## Workshop Context

- `WORKSHOP.md` contains speaker notes for the 45-min session
- Git tags are workshop checkpoints: `step-1-scaffold`, `step-2-data`, `step-3-training`, `step-4-complete`
- Demo output varies per run due to seed=42 shuffling; scorer is deterministic with seed=1
- Rust edition 2024
