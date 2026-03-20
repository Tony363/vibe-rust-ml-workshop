use linfa::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 1;
const SPLIT_RATIO: f32 = 0.8;
const NUM_CLASSES: usize = 3;
const CLASS_NAMES: [&str; NUM_CLASSES] = ["Setosa", "Versicolor", "Virginica"];

fn main() {
    let dataset = linfa_datasets::iris();
    let mut rng = StdRng::seed_from_u64(SEED);
    let dataset = dataset.shuffle(&mut rng);
    let (train, test) = dataset.split_with_ratio(SPLIT_RATIO);

    let predictions = vibe_rust_ml_workshop::build_and_predict(&train, test.records());
    let actuals = test.as_targets();

    let correct = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|(p, a)| p == a)
        .count();
    let total = actuals.len();
    let accuracy = correct as f64 / total as f64;
    let name = vibe_rust_ml_workshop::model_name();

    // Build confusion matrix
    let mut cm = [[0u32; NUM_CLASSES]; NUM_CLASSES];
    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        cm[*actual][*pred] += 1;
    }

    // Per-class metrics
    let mut per_class = Vec::new();
    for c in 0..NUM_CLASSES {
        let tp = cm[c][c] as f64;
        let col_sum: f64 = (0..NUM_CLASSES).map(|r| cm[r][c] as f64).sum();
        let row_sum: f64 = cm[c].iter().map(|&v| v as f64).sum();
        let precision = if col_sum > 0.0 { tp / col_sum } else { 0.0 };
        let recall = if row_sum > 0.0 { tp / row_sum } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        per_class.push((CLASS_NAMES[c], precision, recall, f1, row_sum as u32));
    }

    let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");

    // Build JSON manually
    let cm_json = format!(
        "[[{},{},{}],[{},{},{}],[{},{},{}]]",
        cm[0][0], cm[0][1], cm[0][2], cm[1][0], cm[1][1], cm[1][2], cm[2][0], cm[2][1], cm[2][2]
    );

    let per_class_json: Vec<String> = per_class
        .iter()
        .map(|(name, p, r, f1, s)| {
            format!(
                r#"{{"name":"{name}","precision":{p:.4},"recall":{r:.4},"f1":{f1:.4},"support":{s}}}"#
            )
        })
        .collect();

    println!(
        r#"{{"accuracy":{accuracy:.4},"correct":{correct},"total":{total},"model_name":"{escaped_name}","classes":["Setosa","Versicolor","Virginica"],"confusion_matrix":{cm_json},"per_class":[{pc}]}}"#,
        pc = per_class_json.join(",")
    );
}
