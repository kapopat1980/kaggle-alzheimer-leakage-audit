# Slice-level leakage in the Kaggle Alzheimer MRI benchmark: a scan-level audit

Code, fold definitions and predictions for the paper
"Slice-level leakage inflates deep-learning accuracy on the Kaggle Alzheimer MRI benchmark:
a scan-level audit with grouped cross-validation" (V. Gondalia, K. Popat, Marwadi University).

## Data (not included)
- Kaggle `rajmohan096/alzheimer-mri-preprocessed-dataset` (6,400 images, 128x128)
- Kaggle `ninadaithal/imagesoasis` (OASIS-1 slices; subject ID is in the file name)

## Notebooks (`notebooks/`, run on Kaggle GPU, TensorFlow 2.20)
| Notebook | Purpose |
|---|---|
| experiment2, experiment3, experiment4 | Single 70/15/15 split: fair ablation, improved recipe, fusion, ensembles (Table 4) |
| experiment5a | Leakage audit: 1-NN, near-duplicate groups, tau ladder (Tables 2-3) |
| experiment8d, 8f | Random-split replications (70/15/15 and 60/20/20) |
| experiment10a / 10b | ResNet50 random vs scan-level grouped (tau 0.80) 5-fold CV |
| experiment10e / 10f | Fusion model, same two protocols |
| experiment10g / 10h | ResNet50 grouped at tau 0.95 and tau 0.90 (dose-response) |
| experiment10c / 10d | OASIS-1: random vs subject-grouped 5-fold CV |
| experiment11a-11d | Replicates of 10a-10d with training seed 2042 (11c/11d also use a new 16-slice-per-subject sample); `analysis/rep.py` compares them |

Notebooks named 7a-7f, 8a-8c, 8g, 9a, 9b from earlier iterations are NOT included: they used a
training-data ordering that biased cross-validation and were superseded by the experiment10 series.

## Results (`results/exp10*`)
Per-fold results, fold/group assignments, pooled out-of-fold labels and probabilities, confusion matrices.
`analysis/stats.py` reproduces the pooled metrics and cluster-bootstrap intervals (`final_stats.json`);
`analysis/figs.py` draws the figures. Both scripts expect each results folder as `r10a`, `r10b`, ... (rename `results/exp10a` to `r10a`, etc.) in the working directory.

## Note
GPU training is not bit-wise reproducible; expect small differences when re-running.

## License
MIT (see LICENSE). The datasets keep their own licences on Kaggle.
