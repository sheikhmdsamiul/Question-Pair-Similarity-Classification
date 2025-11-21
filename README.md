# Question-Pair-Similarity-Classification

Project goal
- Classify question pairs as `duplicate` (semantically similar) or `not duplicate` using a Siamese BiLSTM model augmented with engineered features.

Notebooks and what works are done:
- `notebooks/eda_processing_featureeng_baseline.ipynb`
	- Exploratory data analysis (EDA)
	- Text cleaning and preprocessing (Lowercasing,lemmatization, stopword removal)
	- Feature engineering using insights from EDA: lengths, qid frequency, token-overlap ratios, fuzzy-string similarities
	- TF-IDF (1–2 grams) and baseline models for benchmark performence: `LinearSVC`, `RandomForest`
	- Produces processed CSVs (`df_processed_new(1).csv`) to be used for the Simense network.

- `notebooks/siamese_BiLSTM_engineered_features.ipynb`
    - Numeric feature scaling: fits a `StandardScaler` on training numeric features and applies it to test features.
    - fits a Keras `Tokenizer` on the training questions , converts texts to sequences, and pads them to a chosen `MAX_SEQUENCE_LENGTH` based on percentile calculation
	- Builds a Siamese BiLSTM encoder with frozen GloVe(100d) embeddings
	- Model architecture:
        - Shared Bidirectional LSTM encoder (128 units) used as a Siamese encoder for both questions.
        - Interaction features: element-wise subtraction and multiplication of encoded question vectors.
        - Concatenate: q1 encoding, q2 encoding, subtracted, multiplied vectors plus engineered numeric features are concatenated.
        - Classification head: LayerNormalization → Dense(128) → Dropout → LayerNormalization → Dense(64) → Dropout → Dense(1, sigmoid).
        - Embedding layer uses the prebuilt `embedding_matrix` and is set `trainable=False` by default.
	- Trains the model with class weights (computed to address a ~63/37 class imbalance discovered in EDA), callbacks and evaluates with AUC / precision / recall
	- Performs hyperparameter search (KerasTuner) and saves best model (`best_model.keras`)



