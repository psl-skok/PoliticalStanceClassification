# Political Text Classification with TF-IDF and Logistic Regression

This project builds an interpretable political text classification model to distinguish between **Liberal** and **Conservative** Reddit posts using traditional NLP techniques. The focus is on understanding *why* specific modeling choices work, rather than treating machine learning as a black box.

The model uses **TF-IDF features** combined with **Logistic Regression**, demonstrating how careful feature filtering, regularization, and class imbalance handling can meaningfully improve performance on real-world text data.

---

## Dataset

The dataset consists of political Reddit posts labeled by political leaning.

**Preprocessing steps:**
- Removed rows with missing text or labels
- Filtered out very short posts (length ≤ 20 characters)
- Cleaned common text/speach artifacts 

The full dataset and a short, viewable version of the dataset are available in the Data folder of this repository.

---

## Modeling Approach

### Train / Test Split
- 80 / 20 split using `train_test_split`
- **Stratified by class label** to preserve class balance
- Fixed random seed for reproducibility

### Feature Engineering: TF-IDF
Text was vectorized using `TfidfVectorizer` with the following design choices:
- **Unigrams and bigrams (`ngram_range=(1,2)`)** to capture short phrases and context
- **`min_df=2`** to remove rare, noisy terms
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

                  precision    recall  f1-score   support

      Conservative       0.63      0.61      0.62       185
      Liberal            0.76      0.78      0.77       291

      accuracy                               0.71       476
      macro avg          0.69      0.69      0.69       476
      weighted avg       0.71      0.71      0.71       476

## Model Interpretability

One advantage of Logistic Regression is coefficient interpretability. The learned weights reveal meaningful political language patterns.

### Top Conservative-Associated Terms
- libertarian
- free market
- capitalism
- property
- anarcho-capitalism
- russia / ukraine

### Top Liberal-Associated Terms
- social
- democratic
- workers
- union
- progressive
- social democracy
- community

These terms align well with real-world ideological framing, indicating the model is learning coherent semantic signals rather than spurious correlations.

---

## Limitations

- Reddit users are not representative of the broader electorate
- Political ideology is reduced to a binary label
- Language evolves rapidly; models may degrade over time

---

## Future Work

- Add cross-validation and systematic hyperparameter tuning
- Compare against Naive Bayes and linear SVM baselines
- Evaluate ROC-AUC and decision threshold tradeoffs
- Explore transformer-based models for comparison
- Extend to multi-class or ideology spectrum classification

---

## Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook

---

## Project Motivation

This project was built to deepen understanding of applied NLP and machine learning fundamentals while producing a portfolio-quality artifact suitable for technical interviews and data-focused roles.
