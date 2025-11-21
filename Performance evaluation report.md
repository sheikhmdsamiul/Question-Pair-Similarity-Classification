**Performance Evaluation**

This report compares the baseline classical models (TF-IDF + engineered features) with the base Siamese BiLSTM and the tuned Siamese BiLSTM..

Dataset: `train.csv` (Question pairs). Key columns: `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate` (binary target).

Class imbalance: EDA found a class distribution of approximately 63% non-duplicate / 37% duplicate. The Siamese BiLSTM training accounts for this by computing and applying class weights during model.fit to reduce bias toward the majority class.

Key preprocessing: standard text cleaning, lemmatization, engineered lexical features (overlap, fuzzy scores), TF-IDF (1–2 grams) for classical models, engineered lexical features (overlap, fuzzy scores), N(umeric) scaling, Keras Tokenizer + padding, and frozen GloVe(100d) embeddings for the Siamese model.


Comparison:
- Metric improvements from base → tuned Siamese: Loss −0.0247, Accuracy +0.0066, Precision +0.0089, Recall +0.0028, AUC +0.0055. Improvements are consistent and indicate better generalization and ranking after tuning.

- Baseline classical models (LinearSVC, RandomForest with TF-IDF + engineered features). Typical expectations: fast training/inference and competitive precision on surface-similarity cases, but weaker AUC/recall on semantic paraphrases compared to Siamese models.

Baseline numeric results:

- `RandomForest` (TF-IDF + engineered features):
  - Accuracy: 0.825918
  - F1 score: 0.761406
  - Classification report (support: 51009 negative, 29861 positive):
    - class 0 — precision 0.86, recall 0.87, f1 0.86
    - class 1 — precision 0.77, recall 0.75, f1 0.76

- `LinearSVC` (TF-IDF + engineered features):
  - Accuracy: 0.799493
  - F1 score: 0.720059
  - Classification report (support: 51009 negative, 29861 positive):
    - class 0 — precision 0.83, recall 0.86, f1 0.84
    - class 1 — precision 0.74, recall 0.70, f1 0.72

Concise comparison table (key metrics)

| Model | Accuracy | F1 (macro/approx) | Precision (pos) | Recall (pos) | AUC |
|---|---:|---:|---:|---:|---:|
| `RandomForest` | 0.8259 | 0.7614 | 0.77 | 0.75 | N/A |
| `LinearSVC` | 0.7995 | 0.7201 | 0.74 | 0.70 | N/A |
| `Siamese BiLSTM (base)` | 0.8256 | ~0.7945 | 0.7031 | 0.9134 | 0.9220 |
| `Siamese BiLSTM (tuned)` | 0.8322 | ~0.8012 | 0.7120 | 0.9162 | 0.9275 |


Interpretation:
- The tuned Siamese BiLSTM achieves the best overall AUC and highest recall, indicating superior semantic generalization and ranking ability. It slightly improves accuracy and precision over the base Siamese model.

- `RandomForest` produces competitive overall accuracy and F1 — it balances precision and recall better than the Siamese models on the positive class (pos precision 0.77 vs 0.71 for Siamese), but it lags in recall and AUC where semantic understanding matters.

- `LinearSVC` is a solid, lower-cost baseline but underperforms both LinearSVC and Siamese models on F1 and recall.



- **Improvement After Tunining**:
  - Test Loss: 0.3991 → 0.3744 (absolute decrease 0.0247)
  - Test Accuracy: 0.8256 → 0.8322 (absolute increase 0.0066)
  - Test Precision: 0.7031 → 0.7120 (absolute increase 0.0089)
  - Test Recall: 0.9134 → 0.9162 (absolute increase 0.0028)
  - Test AUC: 0.9220 → 0.9275 (absolute increase 0.0055)




**Short summary**
- The best reported model (Siamese BiLSTM + engineered features) demonstrates strong ranking capability (AUC ~ 0.9275) and very high recall (>0.91), with moderate precision (~0.71). Hyperparameter tuning yields measurable improvements.

