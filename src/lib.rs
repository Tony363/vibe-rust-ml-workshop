use linfa::DatasetBase;
use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array1, Array2};

/// Participants: modify this function to build your best model!
/// You can use any linfa algorithm (trees, SVM, KNN, clustering, etc.)
/// but must return predictions as Array1<usize> for the 3 Iris classes.
pub fn build_and_predict(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test_features: &Array2<f64>,
) -> Array1<usize> {
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(4))
        .fit(train)
        .expect("training failed");
    model.predict(test_features)
}

/// Name your model (shown on the leaderboard)
pub fn model_name() -> &'static str {
    "DecisionTree (Gini, depth=4)"
}

// ===================== HARD MODE: Wine Quality =====================
// Red wine quality prediction: 1599 samples, 11 features, 6 classes (scores 3-8).
// Heavily imbalanced and overlapping classes make this much harder than Iris.
// Baseline accuracy: ~57%. Can you break 65%?

/// Hard mode: modify this function to build your best wine quality model!
pub fn build_and_predict_wine(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test_features: &Array2<f64>,
) -> Array1<usize> {
    let model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(4))
        .fit(train)
        .expect("training failed");
    model.predict(test_features)
}

/// Name your wine model (shown on the leaderboard)
pub fn model_name_wine() -> &'static str {
    "DecisionTree (Entropy, depth=4)"
}
