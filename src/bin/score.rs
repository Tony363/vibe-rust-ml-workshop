use linfa::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SEED: u64 = 42;
const SPLIT_RATIO: f32 = 0.8;

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

    println!(
        r#"{{"accuracy":{accuracy:.4},"correct":{correct},"total":{total},"model_name":"{name}"}}"#
    );
}
