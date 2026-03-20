use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array1, Array2};

/// Entropy-based decision tree with shallow depth for better generalization.
pub fn build_and_predict(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test_features: &Array2<f64>,
) -> Array1<usize> {
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(3))
        .fit(train)
        .expect("training failed");
    model.predict(test_features)
}

/// Name your model (shown on the leaderboard)
pub fn model_name() -> &'static str {
    "DecisionTree (Entropy, depth=3)"
}
