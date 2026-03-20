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
    // Train 3 trees with different configs and take majority vote
    let configs: Vec<(SplitQuality, Option<usize>)> = vec![
        (SplitQuality::Entropy, Some(10)),
        (SplitQuality::Gini, Some(5)),
        (SplitQuality::Gini, None),
    ];

    let predictions: Vec<Array1<usize>> = configs
        .into_iter()
        .map(|(quality, depth)| {
            let mut params = DecisionTree::params().split_quality(quality);
            if let Some(d) = depth {
                params = params.max_depth(Some(d));
            }
            let model = params.fit(train).expect("training failed");
            model.predict(test_features)
        })
        .collect();

    // Majority vote across the 3 models
    let n = predictions[0].len();
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let mut votes = [0u32; 3];
                for pred in &predictions {
                    votes[pred[i]] += 1;
                }
                votes
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, v)| *v)
                    .unwrap()
                    .0
            })
            .collect(),
    )
}

/// Name your model (shown on the leaderboard)
pub fn model_name() -> &'static str {
    "Ensemble (3-tree majority vote)"
}
