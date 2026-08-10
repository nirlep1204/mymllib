# `mymllib` Complete API Reference

---

## 1. Preprocessing & Data Utilities (`mymllib.data`)
- `ml.Scaler()`: Standardizes features to zero mean and unit variance.
  - `.fit(X)`, `.transform(X)`, `.fit_transform(X)`
- `ml.Normalizer()`: Min-Max scales features to $[0, 1]$.
  - `.fit(X)`, `.transform(X)`, `.fit_transform(X)`
- `ml.Encoder()`: One-hot encodes categorical column.
  - `.fit(data, col)`, `.transform(data, col)`, `.fit_transform(data, col)`
- `ml.LabelEncoder()`: Converts string/categorical labels into integer indices.
  - `.fit(y)`, `.transform(y)`, `.fit_transform(y)`, `.inverse(y)`
- `ml.split(X, y, test=0.2, seed=None)`: Splits features and labels into train/test sets.
- `ml.xy(data, target)`: Splits DataFrame into feature matrix `X` and target vector `y`.

---

## 2. Supervised Learning Models (`mymllib.models`)

- `ml.Linear(method='gd', lr=0.01, max_iter=1000, reg=None, alpha=0.0)`
  - Methods: `fit(X, y)`, `predict(X)`, `cost(X, y)`
- `ml.Logistic(lr=0.01, max_iter=1000, reg=None, alpha=0.0)`
  - Methods: `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- `ml.Softmax(lr=0.01, max_iter=1000, reg=None, alpha=0.0)`
  - Methods: `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- `ml.GaussianNB()`: Continuous Naive Bayes with Gaussian likelihoods.
  - Methods: `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- `ml.MultinomialNB(alpha=1.0)`: Discrete count feature Naive Bayes.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.BernoulliNB(alpha=1.0, threshold=0.0)`: Binary feature Naive Bayes.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.GDA(shared_cov=True)`: Gaussian Discriminant Analysis (LDA / QDA).
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.KNN(k=5, task='classify')`: K-Nearest Neighbors (Classification & Regression).
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.SVM(C=1.0, kernel='linear', gamma=1.0, degree=3, max_iter=100)`: Support Vector Machine (SMO).
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.SVR(C=1.0, epsilon=0.1, lr=0.01, max_iter=1000)`: Support Vector Regression.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.GLM(family='gaussian', lr=0.01, max_iter=1000)`: Generalized Linear Models (`'gaussian'`, `'bernoulli'`, `'poisson'`).
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.Perceptron(lr=0.01, max_iter=1000)`: Classic Perceptron algorithm.
  - Methods: `fit(X, y)`, `predict(X)`

---

## 3. Clustering & Dimensionality Reduction (`mymllib.cluster`, `mymllib.reduce`)

- `ml.KMeans(k=3, max_iter=100, tol=1e-4, seed=None)`
  - Methods: `fit(X)`, `predict(X)`, `fit_predict(X)`
- `ml.Hierarchical(k=3, linkage='single')`: Linkages: `'single'`, `'complete'`, `'average'`.
  - Methods: `fit(X)`, `fit_predict(X)`
- `ml.GMM(k=3, max_iter=100, tol=1e-4, seed=None)`: Gaussian Mixture Models with EM.
  - Methods: `fit(X)`, `predict(X)`, `predict_proba(X)`
- `ml.PCA(n_components=2)`: Principal Component Analysis.
  - Methods: `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(X)`
- `ml.Factor(n_factors=2, max_iter=100, tol=1e-4)`: Factor Analysis with EM.
  - Methods: `fit(X)`, `transform(X)`, `fit_transform(X)`
- `ml.ICA(n_components=2, max_iter=200, tol=1e-4, seed=None)`: FastICA.
  - Methods: `fit(X)`, `transform(X)`, `fit_transform(X)`

---

## 4. Decision Trees & Ensembles (`mymllib.trees`)

- `ml.Tree(task='classify', max_depth=10, min_samples=2, criterion=None)`: CART Tree.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.Forest(n_trees=100, task='classify', max_depth=10, max_features=None, seed=None)`: Random Forest.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.Bag(base_model, n_models=10, seed=None)`: Bootstrap Aggregation wrapper.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.AdaBoost(n_rounds=50, lr=1.0)`: Multi-class SAMME AdaBoost.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.GradientBoost(n_rounds=100, lr=0.1, max_depth=3, task='regress')`: Gradient Boosting.
  - Methods: `fit(X, y)`, `predict(X)`
- `ml.XGBoost(n_rounds=100, lr=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0, task='regress')`: XGBoost with 2nd-order exact greedy splits.
  - Methods: `fit(X, y)`, `predict(X)`

---

## 5. Neural Networks (`mymllib.nn`)

- `ml.Dense(in_size, out_size, activation='relu')`: Fully-connected layer (`'relu'`, `'sigmoid'`, `'tanh'`, `'softmax'`, `'linear'`).
- Optimizers:
  - `ml.SGD(lr=0.01)`
  - `ml.Momentum(lr=0.01, beta=0.9)`
  - `ml.RMSProp(lr=0.001, beta=0.999)`
  - `ml.Adam(lr=0.001, beta1=0.9, beta2=0.999)`
- `ml.Network(layers)`: Multi-Layer Perceptron.
  - Methods: `fit(X, y, optimizer=None, loss=None, epochs=100, batch_size=32)`, `predict(X)`, `predict_proba(X)`

---

## 6. Metrics & Validation (`mymllib.metrics`, `mymllib.validation`)

- `ml.mse(y, pred)`, `ml.rmse(y, pred)`, `ml.mae(y, pred)`, `ml.r2(y, pred)`
- `ml.accuracy(y, pred)`, `ml.precision(y, pred)`, `ml.recall(y, pred)`, `ml.f1(y, pred)`, `ml.confusion(y, pred)`
- `ml.silhouette(X, labels)`, `ml.wcss(X, labels, centers)`
- `ml.cross_validate(model, X, y, folds=5, metric=None, seed=None)`

---

## 7. Visualization (`mymllib.plot`)

- `ml.scatter(X, y, title=None)`
- `ml.hist(data, col, bins=20)`
- `ml.line(x, y, label=None)`
- `ml.loss_plot(losses)`
- `ml.boundary(model, X, y, res=200)`
- `ml.confusion_plot(y, pred, labels=None)`
- `ml.dendrogram(model)`
- `ml.pca_plot(X, y=None)`
- `ml.cluster_plot(X, labels)`
