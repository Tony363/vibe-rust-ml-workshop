use comfy_table::{Table, presets::UTF8_FULL};
use linfa::prelude::*;

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
}
