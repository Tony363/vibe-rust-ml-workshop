use comfy_table::{Table, presets::UTF8_FULL};
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use linfa_trees::SplitQuality;
use std::time::Instant;

fn main() {
    println!("\n  Vibe Rust ML Workshop — Iris Classification\n");

    // Load the Iris dataset (150 samples, 4 features, 3 classes)
    let dataset = linfa_datasets::iris();

    let feature_names = dataset.feature_names();
    let n_samples = dataset.nsamples();
    let n_features = dataset.nfeatures();

    // Display dataset info
    let mut info_table = Table::new();
    info_table.load_preset(UTF8_FULL);
    info_table.set_header(vec!["Property", "Value"]);
    info_table.add_row(vec!["Dataset", "Iris"]);
    info_table.add_row(vec!["Samples", &n_samples.to_string()]);
    info_table.add_row(vec!["Features", &n_features.to_string()]);
    info_table.add_row(vec!["Classes", "3 (Setosa, Versicolor, Virginica)"]);
    info_table.add_row(vec!["Features", &feature_names.join(", ")]);
    println!("{info_table}\n");

    // Split into training (80%) and testing (20%) sets
    let (train, test) = dataset.split_with_ratio(0.8);

    println!(
        "  Train/Test split: {} training, {} testing samples\n",
        train.nsamples(),
        test.nsamples()
    );

    // --- Model 1: Decision Tree (Gini, max_depth=4) ---
    println!("  Training Model 1: Decision Tree (Gini, depth=4)...");
    let t1 = Instant::now();
    let gini_tree = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(4))
        .fit(&train)
        .expect("Failed to train Gini tree");
    let t1_elapsed = t1.elapsed();
    println!("  -> Trained in {:.2?}", t1_elapsed);

    // --- Model 2: Decision Tree (Entropy, no depth limit) ---
    println!("  Training Model 2: Decision Tree (Entropy, unlimited depth)...");
    let t2 = Instant::now();
    let entropy_tree = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .fit(&train)
        .expect("Failed to train Entropy tree");
    let t2_elapsed = t2.elapsed();
    println!("  -> Trained in {:.2?}\n", t2_elapsed);

    // Suppress unused-variable warnings (used in next step)
    let _ = (&gini_tree, &entropy_tree, &test, &t1_elapsed, &t2_elapsed);
}
