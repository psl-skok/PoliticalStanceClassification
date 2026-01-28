# Political Text Classification with TF-IDF and Logistic Regression

This project builds an interpretable political text classification model to distinguish between **Liberal** and **Conservative** Reddit posts using traditional NLP techniques. The focus is on understanding *why* specific modeling choices work, rather than treating machine learning as a black box.

The model uses **TF-IDF features** combined with **Logistic Regression**, demonstrating how careful feature filtering, regularization, and class imbalance handling can meaningfully improve performance on real-world text data.

---

## Dataset

The dataset consists of political Reddit posts labeled by political leaning.

**Preprocessing steps:**
- Removed rows with missing text or labels
- Filtered out very short posts (length ≤ 20 characters)
- Cleaned common Reddit artifacts (e.g., zero-width Unicode characters)

Due to size and source considerations, the full dataset is not included in this repository.

- A small sample dataset may be provided for reproducibility
- Instructions or a link to the full dataset can be added in `data/README.md`

> **Note:** Reddit data is noisy and not representative of the general population. Results should be interpreted accordingly.

---

## Modeling Approach

### Train / Test Split
- 80 / 20 split using `train_test_split`
- **Stratified by class label** to preserve class balance
- Fixed random seed for reproducibility

### Feature Engineering: TF-IDF
Text was vectorized using `TfidfVectorizer` with the following design choices:
- **Unigrams and bigrams (`ngram_range=(1,2)`)** to capture short phrases and context
- **`min_df=5`** to remove rare, noisy terms
- **`max_df=0.8`** to remove overly common, low-information terms
- **Maximum feature limit** to control dimensionality
- **English stopwords removed** to improve interpretability

These parameters reduce noise, prevent feature explosion, and improve generalization.

---

## Model: Logistic Regression

Logistic Regression was chosen as a strong baseline for high-dimensional, sparse text data.

Key configuration:
- **`class_weight="balanced"`** to compensate for class imbalance
- **Regularization (`C=2.0`)** to balance bias and variance
- **`solver="liblinear"`**, which performs well for binary classification with sparse inputs
- Increased `max_iter` to ensure convergence

Logistic Regression was preferred over Naive Bayes due to:
- Fewer unrealistic independence assumptions
- Better compatibility with TF-IDF features
- Stronger empirical performance with sufficient data
- Better-calibrated probability estimates

---

## Results

**Test Set Performance:**

- Accuracy: ~71%
- Balanced precision and recall across classes

