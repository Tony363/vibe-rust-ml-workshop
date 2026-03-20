use linfa::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SEED: u64 = 42;
const SPLIT_RATIO: f32 = 0.8;
const CLASS_NAMES: [&str; 3] = ["Setosa", "Versicolor", "Virginica"];
const N_CLASSES: usize = 3;

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

    // Build confusion matrix: cm[actual][predicted]
    let mut cm = [[0u32; N_CLASSES]; N_CLASSES];
    for (p, a) in predictions.iter().zip(actuals.iter()) {
        if *a < N_CLASSES && *p < N_CLASSES {
            cm[*a][*p] += 1;
        }
    }

    // Per-class stats
    let mut per_class = Vec::new();
    for c in 0..N_CLASSES {
        let class_total: u32 = cm[c].iter().sum();
        let class_correct = cm[c][c];
        per_class.push((CLASS_NAMES[c], class_correct, class_total));
    }

    // Format confusion matrix as JSON array of arrays
    let cm_json: Vec<String> = cm
        .iter()
        .map(|row| {
            let cells: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            format!("[{}]", cells.join(","))
        })
        .collect();

    // Format per_class as JSON array of objects
    let pc_json: Vec<String> = per_class
        .iter()
        .map(|(n, c, t)| format!(r#"{{"name":"{n}","correct":{c},"total":{t}}}"#))
        .collect();

    println!(
        r#"{{"accuracy":{accuracy:.4},"correct":{correct},"total":{total},"model_name":"{name}","confusion_matrix":[{cm}],"per_class":[{pc}]}}"#,
        cm = cm_json.join(","),
        pc = pc_json.join(","),
    );
}
