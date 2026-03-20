use linfa::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 1;
const SPLIT_RATIO: f32 = 0.8;

fn main() {
    let dataset = linfa_datasets::winequality();
    let mut rng = StdRng::seed_from_u64(SEED);
    let dataset = dataset.shuffle(&mut rng);
    let (train, test) = dataset.split_with_ratio(SPLIT_RATIO);

    let predictions = vibe_rust_ml_workshop::build_and_predict_wine(&train, test.records());
    let actuals = test.as_targets();

    let correct = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|(p, a)| p == a)
        .count();
    let total = actuals.len();
    let accuracy = correct as f64 / total as f64;
    let name = vibe_rust_ml_workshop::model_name_wine();

    // Discover classes from the data
    let mut class_set = std::collections::BTreeSet::new();
    for &t in actuals.iter() {
        class_set.insert(t);
    }
    for &t in predictions.iter() {
        class_set.insert(t);
    }
    let classes: Vec<usize> = class_set.into_iter().collect();
    let num_classes = classes.len();

    // Build confusion matrix (dynamic size)
    let mut cm = vec![vec![0u32; num_classes]; num_classes];
    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        let ai = classes.iter().position(|&c| c == *actual).unwrap();
        let pi = classes.iter().position(|&c| c == *pred).unwrap();
        cm[ai][pi] += 1;
    }

    // Per-class metrics
    let mut per_class = Vec::new();
    for (i, &class_label) in classes.iter().enumerate() {
        let tp = cm[i][i] as f64;
        let col_sum: f64 = (0..num_classes).map(|r| cm[r][i] as f64).sum();
        let row_sum: f64 = cm[i].iter().map(|&v| v as f64).sum();
        let precision = if col_sum > 0.0 { tp / col_sum } else { 0.0 };
        let recall = if row_sum > 0.0 { tp / row_sum } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        per_class.push((class_label, precision, recall, f1, row_sum as u32));
    }

    let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");

    // Build JSON
    let classes_json: Vec<String> = classes.iter().map(|c| format!("\"Quality {c}\"")).collect();
    let cm_json = format!(
        "[{}]",
        cm.iter()
            .map(|row| format!(
                "[{}]",
                row.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let per_class_json: Vec<String> = per_class
        .iter()
        .map(|(label, p, r, f1, s)| {
            format!(
                r#"{{"name":"Quality {label}","precision":{p:.4},"recall":{r:.4},"f1":{f1:.4},"support":{s}}}"#
            )
        })
        .collect();

    println!(
        r#"{{"accuracy":{accuracy:.4},"correct":{correct},"total":{total},"model_name":"{escaped_name}","classes":[{classes}],"confusion_matrix":{cm_json},"per_class":[{pc}]}}"#,
        classes = classes_json.join(","),
        pc = per_class_json.join(",")
    );
}
