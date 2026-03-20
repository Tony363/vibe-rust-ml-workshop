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
        .max_depth(Some(8))
        .fit(train)
        .expect("training failed");
    model.predict(test_features)
}

/// Name your model (shown on the leaderboard)
pub fn model_name() -> &'static str {
    "DecisionTree (Gini, depth=8)"
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
    // Normalise features to 0-1 using min/max from training data
    let train_features = train.records();
    let n_features = train_features.ncols();
    let mins: Vec<f64> = (0..n_features).map(|i| train_features.column(i).fold(f64::INFINITY, |a, &b| f64::min(a, b))).collect();
    let maxs: Vec<f64> = (0..n_features).map(|i| train_features.column(i).fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b))).collect();

    let normalise = |data: &Array2<f64>| -> Array2<f64> {
        let mut out = data.clone();
        for i in 0..n_features {
            let range = maxs[i] - mins[i];
            if range > 0.0 {
                out.column_mut(i).mapv_inplace(|v| (v - mins[i]) / range);
            }
        }
        out
    };

    let norm_train_features = normalise(train_features);
    let norm_train = DatasetBase::new(norm_train_features, train.targets().to_owned());
    let norm_test = normalise(test_features);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(20))
        .fit(&norm_train)
        .expect("training failed");
    model.predict(&norm_test)
}

/// Name your wine model (shown on the leaderboard)
pub fn model_name_wine() -> &'static str {
    "DecisionTree (Entropy, depth=20, normalised)"
}
