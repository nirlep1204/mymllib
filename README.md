# mymllib

A machine learning library I'm building from scratch using Python and NumPy.

---

## Why I built this

When I first started learning ML, I'd call `model.fit()` in scikit-learn and get a result — but I had no idea what was actually happening inside. The gradients, the loss surfaces, the convergence behavior — it was all hidden behind a clean API.

That bothered me. So I decided to build my own library from the ground up. No scikit-learn, no PyTorch, no TensorFlow — just NumPy and the math I derived on paper.

What started as a weekend project turned into something much bigger. I kept adding modules — preprocessing, metrics, cross-validation, visualization, ensemble methods — until it became a full-fledged library.

This project is still actively evolving. I keep adding new algorithms and improving existing ones as I learn more.

---

## Quick Example

```python
import mymllib as ml
import pandas as pd

# load and prep data
df = pd.read_csv('data.csv')
X, y = ml.xy(df, target='label')
X_train, X_test, y_train, y_test = ml.split(X, y, test_size=0.2)

scaler = ml.Scaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# train
model = ml.Forest(n_trees=50, task='classify')
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
print("Accuracy:", ml.accuracy(y_test, preds))

# need help?
ml.help('forest')
```

---

## What's inside

### Preprocessing
`Scaler`, `Normalizer`, `Encoder`, `LabelEncoder`, `split`, `xy`

### Supervised Learning
- **Linear Models**: `Linear` (GD + Normal Equation, L1/L2 regularization)
- **Classification**: `Logistic`, `Softmax`, `Perceptron`
- **Probabilistic**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `GDA`
- **Instance-based**: `KNN`
- **Kernel Methods**: `SVM` (Linear/RBF/Poly via SMO), `SVR`
- **GLMs**: `GLM` (Gaussian, Bernoulli, Poisson)

### Unsupervised Learning
- **Clustering**: `KMeans`, `Hierarchical`, `GMM`
- **Dimensionality Reduction**: `PCA`, `Factor`, `ICA`

### Trees & Ensembles
- `Tree` (CART with Gini/Entropy)
- `Forest` (Random Forest), `Bag` (Bagging)
- `AdaBoost`, `GradientBoost`, `XGBoost`

### Neural Networks
- `Network` + `Dense` layers with manual backprop
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Optimizers: SGD, Momentum, RMSProp, Adam

### Evaluation & Tuning
- **Metrics**: `accuracy`, `f1`, `mse`, `r2`, `roc_auc_score`, `silhouette`, etc.
- **Cross-Validation**: `KFold`, `StratifiedKFold`, `LeaveOneOut`, `ShuffleSplit`, `TimeSeriesSplit`
- **Search**: `GridSearchCV`, `RandomSearchCV`

### Visualization
`scatter`, `line`, `loss_plot`, `boundary`, `roc_plot`, `confusion_plot`, `dendrogram`, `pca_plot`, `cluster_plot`, `residual_plot`, `feature_importance`

### Built-in Help
```python
ml.help()           # list everything
ml.help('svm')      # detailed docs for SVM
ml.help('kfold')    # cross-validation usage
```

---

## Empirical Complexity Benchmarks

I was curious: how do my from-scratch implementations actually scale compared to what the textbooks say?

So I wrote a small benchmarking script (`benchmarks/benchmark.py`) that times each algorithm across increasing sample sizes. A few things surprised me:

- **Linear Regression** barely changed between N=100 and N=500. NumPy's C-level BLAS operations are so optimized that the theoretical O(Nd²) complexity is completely masked at small scales.
- **SVM (SMO solver)** scaled the worst of anything I tested — going from N=100 to N=500, training time grew by ~22x. That's well below the ~125x a naive O(N³) bound would predict for a 5x increase in N, which makes sense once you factor in how SMO actually works: it optimizes one pair of Lagrange multipliers at a time using a working-set heuristic, rather than touching every pairwise constraint on every iteration. So the real-world growth curve is steep, but softer than the textbook worst case. Still the clearest complexity wall I hit in the whole library.
- **Decision Trees** scaled noticeably but gracefully (~6x growth), which lines up with the O(N log N) expectation.

The full plot is saved in `benchmarks/complexity_plot.png`.

---

## Project Structure

```text
mymllib/
├── mymllib/            # the library itself
│   ├── models/         # supervised learning algorithms
│   ├── trees/          # decision trees and ensembles
│   ├── cluster/        # unsupervised clustering
│   ├── reduce/         # dimensionality reduction (PCA, ICA, Factor)
│   ├── nn/             # neural network framework
│   ├── metrics/        # evaluation metrics
│   ├── validation/     # cross-validation and hyperparameter search
│   ├── plot/           # visualization utilities
│   ├── data/           # preprocessing (scalers, encoders, splitting)
│   └── help.py         # interactive help system
├── benchmarks/         # empirical complexity analysis
├── docs/               # API reference
└── setup.py
```

---

## Setup

```bash
git clone https://github.com/nirlep1204/mymllib.git
cd mymllib
pip install -e .

# run the benchmarks (default or custom sample sizes)
python benchmarks/benchmark.py
python benchmarks/benchmark.py --sizes 100 200 500 1000
```

---

## Roadmap

This library is still a work in progress. Here's what I'm planning next:

- [ ] Convolutional layers for image tasks
- [ ] Transformer self-attention mechanism
- [ ] Dropout and BatchNorm regularization
- [ ] `pip install mymllib` (PyPI packaging)
- [ ] Interactive documentation website
- [ ] More examples and tutorials

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Built by **Nirlep Makwana**
