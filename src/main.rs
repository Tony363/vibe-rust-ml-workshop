use comfy_table::{presets::UTF8_FULL, Table};
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use linfa_trees::SplitQuality;
use ndarray::Axis;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;
use vibe_rust_ml_workshop::{build_and_predict, model_name};

const RNG_SEED: u64 = 42;

const CLASS_NAMES: [&str; 3] = ["Setosa", "Versicolor", "Virginica"];

fn main() {
    println!("\n  Vibe Rust ML Workshop -- Iris Classification\n");

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
    info_table.add_row(vec!["Feature Names", &feature_names.join(", ")]);
    println!("{info_table}\n");

    // Shuffle and split into training (80%) and testing (20%) sets
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dataset = dataset.shuffle(&mut rng);
    let (train, test) = dataset.split_with_ratio(0.8);
    println!(
        "  Train/Test split: {} training, {} testing samples\n",
        train.nsamples(),
        test.nsamples()
    );

    // --- Model 1: Decision Tree (Gini, max_depth=4) via lib.rs ---
    println!("  Training Model 1: {}...", model_name());
    let t1 = Instant::now();
    let gini_pred = build_and_predict(&train, test.records());
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

    // ==================== EVALUATION ====================

    // Model 1 predictions already computed via build_and_predict
    let entropy_pred = entropy_tree.predict(&test);

    // Confusion matrices via linfa (entropy model)
    let entropy_cm = entropy_pred.confusion_matrix(&test).expect("confusion matrix");
    let entropy_acc = entropy_cm.accuracy();

    // Compute gini accuracy manually (gini_pred is Array1<usize> from lib.rs)
    let actuals: Vec<usize> = test.as_targets().iter().copied().collect();
    let gini_preds: Vec<usize> = gini_pred.iter().copied().collect();
    let gini_correct = gini_preds.iter().zip(actuals.iter()).filter(|(p, a)| p == a).count();
    let gini_acc = gini_correct as f64 / actuals.len() as f64;

    // --- Model Comparison Table ---
    let mut cmp_table = Table::new();
    cmp_table.load_preset(UTF8_FULL);
    cmp_table.set_header(vec![
        "Model",
        "Split Quality",
        "Max Depth",
        "Accuracy",
        "Train Time",
    ]);
    cmp_table.add_row(vec![
        "Tree 1".to_string(),
        "Gini".to_string(),
        "4".to_string(),
        format!("{:.1}%", gini_acc * 100.0),
        format!("{:.2?}", t1_elapsed),
    ]);
    cmp_table.add_row(vec![
        "Tree 2".to_string(),
        "Entropy".to_string(),
        "None".to_string(),
        format!("{:.1}%", entropy_acc * 100.0),
        format!("{:.2?}", t2_elapsed),
    ]);
    println!("  Model Comparison");
    println!("{cmp_table}\n");

    // --- Confusion Matrix (Gini model) ---
    println!("  Confusion Matrix (Tree 1 -- Gini)");
    print_confusion_matrix(&actuals, &gini_preds);

    // --- Sample Predictions Table (first 10) ---
    println!("\n  Sample Predictions (first 10 test samples)");
    let mut pred_table = Table::new();
    pred_table.load_preset(UTF8_FULL);
    pred_table.set_header(vec![
        "#", "Sepal L", "Sepal W", "Petal L", "Petal W", "Actual", "Predicted",
    ]);

    let records = test.records();
    let n_show = 10.min(test.nsamples());

    for i in 0..n_show {
        let row = records.index_axis(Axis(0), i);
        let actual = actuals[i];
        let predicted = gini_preds[i];
        let marker = if actual == predicted { "" } else { " [X]" };
        pred_table.add_row(vec![
            format!("{}", i + 1),
            format!("{:.1}", row[0]),
            format!("{:.1}", row[1]),
            format!("{:.1}", row[2]),
            format!("{:.1}", row[3]),
            CLASS_NAMES[actual].to_string(),
            format!("{}{}", CLASS_NAMES[predicted], marker),
        ]);
    }
    println!("{pred_table}\n");

    // Machine-readable score line for CI leaderboard
    let best_acc = gini_acc.max(entropy_acc);
    println!("LEADERBOARD_SCORE best={:.4} gini={:.4} entropy={:.4}", best_acc, gini_acc, entropy_acc);
}

fn print_confusion_matrix(actuals: &[usize], predictions: &[usize]) {
    let n_classes = CLASS_NAMES.len();
    let mut matrix = vec![vec![0usize; n_classes]; n_classes];

    for (&actual, &predicted) in actuals.iter().zip(predictions.iter()) {
        if actual < n_classes && predicted < n_classes {
            matrix[actual][predicted] += 1;
        }
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);

    let mut header: Vec<String> = vec!["Actual \\ Predicted".to_string()];
    for name in &CLASS_NAMES {
        header.push(name.to_string());
    }
    table.set_header(header);

    for (i, row) in matrix.iter().enumerate() {
        let mut table_row = vec![CLASS_NAMES[i].to_string()];
        for val in row {
            table_row.push(val.to_string());
        }
        table.add_row(table_row);
    }

    println!("{table}");
}
