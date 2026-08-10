"""Built-in help system for mymllib."""

_HELP = {
    # Preprocessing
    "scaler": {
        "syntax": "ml.Scaler()",
        "description": "Standardize features to zero mean and unit variance: (x - mean) / std",
        "category": "Preprocessing",
        "example": "scaler = ml.Scaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)",
        "params": {},
        "methods": {
            "fit(X)": "Compute mean and std from training data for later scaling.",
            "transform(X)": "Perform standardization on X using fitted mean and std.",
            "fit_transform(X)": "Fit to X, then return standardized features.",
            "inverse_transform(X)": "Reconstruct original feature values from standardized data."
        }
    },
    "normalizer": {
        "syntax": "ml.Normalizer()",
        "description": "Min-Max feature scaling to range [0, 1]: (x - min) / (max - min)",
        "category": "Preprocessing",
        "example": "norm = ml.Normalizer()\nX_train_norm = norm.fit_transform(X_train)\nX_test_norm = norm.transform(X_test)",
        "params": {},
        "methods": {
            "fit(X)": "Compute min and max from training data for scaling.",
            "transform(X)": "Scale features of X to range [0, 1] using fitted min and max.",
            "fit_transform(X)": "Fit to X, then return normalized features.",
            "inverse_transform(X)": "Reconstruct original feature values from normalized data."
        }
    },
    "encoder": {
        "syntax": "ml.Encoder()",
        "description": "One-hot encode categorical column into binary indicator columns",
        "category": "Preprocessing",
        "example": "enc = ml.Encoder()\ndf_enc = enc.fit_transform(df, 'gender')",
        "params": {},
        "methods": {
            "fit(data, col)": "Discover unique categories present in column.",
            "transform(data, col)": "Transform column in DataFrame into one-hot binary columns.",
            "fit_transform(data, col)": "Fit to column and return one-hot encoded DataFrame."
        }
    },
    "labelencoder": {
        "syntax": "ml.LabelEncoder()",
        "description": "Convert discrete/string labels to integers [0, 1, ..., K-1]",
        "category": "Preprocessing",
        "example": "le = ml.LabelEncoder()\ny_encoded = le.fit_transform(y)\ny_orig = le.inverse(y_encoded)",
        "params": {},
        "methods": {
            "fit(y)": "Fit label encoder to unique classes.",
            "transform(y)": "Transform labels to normalized integer encoding.",
            "fit_transform(y)": "Fit label encoder and return integer array.",
            "inverse(y_encoded)": "Transform integer labels back to original class names."
        }
    },
    "split": {
        "syntax": "ml.split(X, y, test=0.2, seed=None)",
        "description": "Split features X and target y into random train and test subsets",
        "category": "Preprocessing",
        "example": "X_train, X_test, y_train, y_test = ml.split(X, y, test=0.25, seed=42)",
        "params": {
            "X": "DataFrame or 2D array of shape (n_samples, n_features)",
            "y": "Series or 1D array of shape (n_samples,)",
            "test": "float in range (0.0, 1.0) (e.g. 0.2, 0.25, 0.33) [default: 0.2] - Fraction of data for test set",
            "seed": "int >= 0 or None [default: None] - Random seed for reproducible shuffling"
        }
    },
    "xy": {
        "syntax": "ml.xy(data, target)",
        "description": "Split a pandas DataFrame into feature matrix X and target vector y",
        "category": "Preprocessing",
        "example": "X, y = ml.xy(df, 'price')",
        "params": {
            "data": "pandas DataFrame containing features and target column",
            "target": "str - Name of the target column to extract as y"
        }
    },

    # Supervised Models
    "linear": {
        "syntax": "ml.Linear(method='gd', lr=0.01, max_iter=1000, batch_size=None, reg=None, alpha=0.0, seed=None)",
        "description": "Linear Regression with Gradient Descent or Closed-form Normal Equation",
        "category": "Supervised Models",
        "example": "# Normal Equation (Closed-form):\nmodel = ml.Linear(method='normal')\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)\n\n# Gradient Descent (Batch / SGD / Mini-Batch):\nmodel_gd = ml.Linear(method='gd', lr=0.01, max_iter=500, batch_size=32)\nmodel_gd.fit(X_train, y_train)",
        "params": {
            "method": "Options: 'gd' (Gradient Descent) | 'normal' or 'closed' (Normal Equation) [default: 'gd']",
            "lr": "float > 0.0 (e.g. 0.01, 0.001, 0.0001) [default: 0.01] - Learning rate for gradient descent",
            "max_iter": "int >= 1 (e.g. 100, 500, 1000) [default: 1000] - Number of iterations/epochs",
            "batch_size": "Options: None (Full Batch GD) | 1 (Stochastic GD / SGD) | int >= 1 (e.g. 16, 32, 64 for Mini-Batch) [default: None]",
            "reg": "Options: None (No penalty) | 'l2' (Ridge regression) | 'l1' (Lasso regression) [default: None]",
            "alpha": "float >= 0.0 (e.g. 0.0, 0.1, 1.0) [default: 0.0] - Regularization penalty multiplier",
            "seed": "int >= 0 or None [default: None] - Random seed for batch shuffling"
        },
        "methods": {
            "fit(X, y)": "Fit linear model weights.",
            "predict(X)": "Predict continuous values for samples in X.",
            "cost(X, y)": "Compute Mean Squared Error loss on X and y."
        }
    },
    "logistic": {
        "syntax": "ml.Logistic(lr=0.01, max_iter=1000, batch_size=None, reg=None, alpha=0.0, seed=None)",
        "description": "Logistic Regression for binary classification with Sigmoid link",
        "category": "Supervised Models",
        "example": "model = ml.Logistic(lr=0.05, max_iter=300)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)\nprobs = model.predict_proba(X_test)",
        "params": {
            "lr": "float > 0.0 (e.g. 0.01, 0.1) [default: 0.01] - Learning rate",
            "max_iter": "int >= 1 [default: 1000] - Number of iterations/epochs",
            "batch_size": "Options: None (Full Batch) | 1 (Stochastic GD / SGD) | int >= 1 (e.g. 32 for Mini-Batch) [default: None]",
            "reg": "Options: None | 'l2' (Ridge) | 'l1' (Lasso) [default: None]",
            "alpha": "float >= 0.0 [default: 0.0] - Regularization strength",
            "seed": "int >= 0 or None [default: None] - Random seed for shuffling"
        },
        "methods": {
            "fit(X, y)": "Fit logistic regression weights.",
            "predict(X)": "Predict binary class labels {0, 1}.",
            "predict_proba(X)": "Predict probability of positive class P(y=1|x) in [0, 1].",
            "cost(X, y)": "Compute binary cross-entropy loss."
        }
    },
    "softmax": {
        "syntax": "ml.Softmax(lr=0.01, max_iter=1000, batch_size=None, seed=None)",
        "description": "Softmax Regression (Multinomial Logistic Regression) for multi-class classification",
        "category": "Supervised Models",
        "example": "model = ml.Softmax(lr=0.05, max_iter=400)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)\nprobs = model.predict_proba(X_test)",
        "params": {
            "lr": "float > 0.0 [default: 0.01] - Learning rate",
            "max_iter": "int >= 1 [default: 1000] - Number of iterations/epochs",
            "batch_size": "Options: None (Full Batch) | 1 (SGD) | int >= 1 (e.g. 32 for Mini-Batch) [default: None]",
            "seed": "int >= 0 or None [default: None] - Random seed for shuffling"
        },
        "methods": {
            "fit(X, y)": "Fit softmax weight matrix.",
            "predict(X)": "Predict discrete multi-class labels.",
            "predict_proba(X)": "Predict class probability distribution of shape (n_samples, n_classes)."
        }
    },
    "gaussiannb": {
        "syntax": "ml.GaussianNB()",
        "description": "Gaussian Naive Bayes for continuous features using normal probability density",
        "category": "Supervised Models",
        "example": "model = ml.GaussianNB()\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {},
        "methods": {
            "fit(X, y)": "Fit class priors, means, and variances from training data.",
            "predict(X)": "Predict class labels for X.",
            "predict_proba(X)": "Predict class posterior probabilities."
        },
        "variants": ["GaussianNB", "MultinomialNB", "BernoulliNB"]
    },
    "multinomialnb": {
        "syntax": "ml.MultinomialNB(alpha=1.0)",
        "description": "Multinomial Naive Bayes for discrete counts and text frequencies",
        "category": "Supervised Models",
        "example": "model = ml.MultinomialNB(alpha=1.0)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {
            "alpha": "float >= 0.0 (e.g. 1.0 for Laplace, 0.1 for Lidstone) [default: 1.0] - Additive smoothing parameter"
        },
        "methods": {
            "fit(X, y)": "Fit log feature probabilities per class.",
            "predict(X)": "Predict class labels for discrete feature counts."
        }
    },
    "bernoullinb": {
        "syntax": "ml.BernoulliNB(alpha=1.0, threshold=0.0)",
        "description": "Bernoulli Naive Bayes for binary / boolean feature data",
        "category": "Supervised Models",
        "example": "model = ml.BernoulliNB(alpha=1.0, threshold=0.0)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {
            "alpha": "float >= 0.0 [default: 1.0] - Additive Laplace smoothing parameter",
            "threshold": "float [default: 0.0] - Threshold for binarizing continuous features (x > threshold -> 1)"
        },
        "methods": {
            "fit(X, y)": "Fit Bernoulli feature probabilities per class.",
            "predict(X)": "Predict class labels."
        }
    },
    "naivebayes": {
        "group": True,
        "variants": ["GaussianNB", "MultinomialNB", "BernoulliNB"]
    },
    "gda": {
        "syntax": "ml.GDA(shared_cov=True)",
        "description": "Gaussian Discriminant Analysis (LDA with shared covariance; QDA with separate covariance)",
        "category": "Supervised Models",
        "example": "model = ml.GDA(shared_cov=True) # LDA\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {
            "shared_cov": "bool True or False [default: True] - If True: Linear Discriminant Analysis (tied covariance). If False: Quadratic Discriminant Analysis (class-specific covariance)."
        },
        "methods": {
            "fit(X, y)": "Fit class priors, means, and covariance matrices.",
            "predict(X)": "Predict class labels using maximum discriminant score."
        }
    },
    "knn": {
        "syntax": "ml.KNN(k=5, task='classify')",
        "description": "K-Nearest Neighbors for classification (majority voting) or regression (mean value)",
        "category": "Supervised Models",
        "example": "# Classification:\nknn = ml.KNN(k=5, task='classify')\nknn.fit(X_train, y_train)\npreds = knn.predict(X_test)\n\n# Regression:\nknn_reg = ml.KNN(k=3, task='regress')\nknn_reg.fit(X_train, y_train)",
        "params": {
            "k": "int >= 1 (e.g. 1, 3, 5, 7, 11) [default: 5] - Number of nearest neighbors to query",
            "task": "Options: 'classify' (majority vote) | 'regress' (average value) [default: 'classify']"
        },
        "methods": {
            "fit(X, y)": "Store training instances for nearest neighbor search.",
            "predict(X)": "Predict labels or values for query samples."
        }
    },
    "svm": {
        "syntax": "ml.SVM(C=1.0, kernel='linear', gamma=1.0, degree=3, max_iter=100)",
        "description": "Support Vector Machine classifier with Sequential Minimal Optimization (SMO)",
        "category": "Supervised Models",
        "example": "model = ml.SVM(C=1.0, kernel='rbf', gamma=0.5)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {
            "C": "float > 0.0 (e.g. 0.1, 1.0, 10.0) [default: 1.0] - Regularization parameter / slack penalty",
            "kernel": "Options: 'linear' | 'rbf' (Gaussian kernel) | 'poly' (Polynomial kernel) [default: 'linear']",
            "gamma": "float > 0.0 (e.g. 0.1, 0.5, 1.0) [default: 1.0] - Kernel coefficient for 'rbf' and 'poly'",
            "degree": "int >= 1 (e.g. 2, 3, 4) [default: 3] - Degree of the polynomial kernel function",
            "max_iter": "int >= 1 [default: 100] - Maximum SMO optimization passes without alpha changes"
        },
        "methods": {
            "fit(X, y)": "Train SVM dual multipliers (alphas) and intercept.",
            "predict(X)": "Predict class labels for samples in X."
        }
    },
    "svr": {
        "syntax": "ml.SVR(C=1.0, epsilon=0.1, lr=0.01, max_iter=1000)",
        "description": "Support Vector Regression with epsilon-insensitive loss",
        "category": "Supervised Models",
        "example": "svr = ml.SVR(C=1.0, epsilon=0.1, lr=0.01)\nsvr.fit(X_train, y_train)\npreds = svr.predict(X_test)",
        "params": {
            "C": "float > 0.0 [default: 1.0] - Regularization penalty",
            "epsilon": "float >= 0.0 (e.g. 0.0, 0.05, 0.1, 0.2) [default: 0.1] - Epsilon tube width where errors are ignored",
            "lr": "float > 0.0 [default: 0.01] - Learning rate",
            "max_iter": "int >= 1 [default: 1000] - Maximum training iterations"
        },
        "methods": {
            "fit(X, y)": "Fit SVR support vectors and weights.",
            "predict(X)": "Predict continuous values for samples in X."
        }
    },
    "glm": {
        "syntax": "ml.GLM(family='gaussian', lr=0.01, max_iter=1000)",
        "description": "Generalized Linear Models connecting linear predictors via exponential link families",
        "category": "Supervised Models",
        "example": "glm = ml.GLM(family='poisson', lr=0.01)\nglm.fit(X_train, y_train)\npreds = glm.predict(X_test)",
        "params": {
            "family": "Options: 'gaussian' (identity link for continuous regression) | 'bernoulli' (logit link for binary classification) | 'poisson' (log link for count data) [default: 'gaussian']",
            "lr": "float > 0.0 [default: 0.01] - Gradient descent learning rate",
            "max_iter": "int >= 1 [default: 1000] - Number of optimization iterations"
        },
        "methods": {
            "fit(X, y)": "Fit GLM parameters using maximum likelihood gradients.",
            "predict(X)": "Predict expected values or classes for X."
        }
    },
    "perceptron": {
        "syntax": "ml.Perceptron(lr=0.01, max_iter=1000)",
        "description": "Classic Perceptron linear binary classification algorithm",
        "category": "Supervised Models",
        "example": "model = ml.Perceptron(lr=0.01, max_iter=200)\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)",
        "params": {
            "lr": "float > 0.0 [default: 0.01] - Learning rate step size",
            "max_iter": "int >= 1 [default: 1000] - Maximum number of passes over the dataset"
        },
        "methods": {
            "fit(X, y)": "Fit perceptron decision boundary.",
            "predict(X)": "Predict binary labels {0, 1} for samples in X."
        }
    },

    # Unsupervised Models
    "kmeans": {
        "syntax": "ml.KMeans(k=3, max_iter=100, tol=1e-4, seed=None)",
        "description": "K-Means clustering algorithm using Lloyd's Expectation-Maximization",
        "category": "Unsupervised Models",
        "example": "km = ml.KMeans(k=3, seed=42)\nkm.fit(X)\nlabels = km.predict(X)\ncenters = km.centers",
        "params": {
            "k": "int >= 1 (e.g. 2, 3, 5, 8) [default: 3] - Number of clusters to form",
            "max_iter": "int >= 1 [default: 100] - Maximum iterations for single run",
            "tol": "float > 0.0 [default: 1e-4] - Relative tolerance to declare convergence",
            "seed": "int >= 0 or None [default: None] - Random seed for centroid initialization"
        },
        "methods": {
            "fit(X)": "Compute k-means clustering on X.",
            "predict(X)": "Predict closest cluster each sample in X belongs to.",
            "fit_predict(X)": "Compute clusters and return cluster index labels."
        }
    },
    "hierarchical": {
        "syntax": "ml.Hierarchical(k=3, linkage='single')",
        "description": "Agglomerative Hierarchical Clustering with bottom-up distance merging",
        "category": "Unsupervised Models",
        "example": "hc = ml.Hierarchical(k=3, linkage='complete')\nlabels = hc.fit_predict(X)",
        "params": {
            "k": "int >= 1 [default: 3] - The target number of clusters to find",
            "linkage": "Options: 'single' (minimum pairwise distance) | 'complete' (maximum pairwise distance) | 'average' (mean distance) [default: 'single']"
        },
        "methods": {
            "fit(X)": "Perform agglomerative clustering on X.",
            "fit_predict(X)": "Fit and return cluster label assignments."
        }
    },
    "gmm": {
        "syntax": "ml.GMM(k=3, max_iter=100, tol=1e-4, seed=None)",
        "description": "Gaussian Mixture Model with soft-assignment Expectation-Maximization (EM)",
        "category": "Unsupervised Models",
        "example": "gmm = ml.GMM(k=3, seed=42)\ngmm.fit(X)\nlabels = gmm.predict(X)\nprobs = gmm.predict_proba(X)",
        "params": {
            "k": "int >= 1 [default: 3] - Number of Gaussian mixture components",
            "max_iter": "int >= 1 [default: 100] - Maximum EM iterations",
            "tol": "float > 0.0 [default: 1e-4] - Log-likelihood convergence tolerance",
            "seed": "int >= 0 or None [default: None] - Random seed for initialization"
        },
        "methods": {
            "fit(X)": "Estimate model parameters (weights, means, covariances) with EM.",
            "predict(X)": "Predict hardest component cluster index for each sample.",
            "predict_proba(X)": "Compute posterior probability responsibilities of shape (n_samples, k)."
        }
    },

    # Dimensionality Reduction
    "pca": {
        "syntax": "ml.PCA(n_components=2)",
        "description": "Principal Component Analysis via eigendecomposition of covariance matrix",
        "category": "Dimensionality Reduction",
        "example": "pca = ml.PCA(n_components=2)\nX_proj = pca.fit_transform(X)\nprint('Variance Ratios:', pca.explained_variance_ratio)",
        "params": {
            "n_components": "int in range [1, n_features] (e.g. 1, 2, 3) [default: 2] - Number of principal components to keep"
        },
        "methods": {
            "fit(X)": "Fit PCA on X by calculating eigenvectors and eigenvalues.",
            "transform(X)": "Apply dimensionality reduction to X.",
            "fit_transform(X)": "Fit model with X and apply dimensionality reduction on X.",
            "inverse_transform(X_proj)": "Transform projected data back to original space."
        }
    },
    "factor": {
        "syntax": "ml.Factor(n_factors=2, max_iter=100, tol=1e-4)",
        "description": "Factor Analysis via Expectation-Maximization for latent variable discovery",
        "category": "Dimensionality Reduction",
        "example": "fa = ml.Factor(n_factors=2)\nX_latent = fa.fit_transform(X)",
        "params": {
            "n_factors": "int in range [1, n_features] [default: 2] - Number of latent factors to extract",
            "max_iter": "int >= 1 [default: 100] - Maximum EM iterations",
            "tol": "float > 0.0 [default: 1e-4] - Log-likelihood convergence threshold"
        },
        "methods": {
            "fit(X)": "Fit factor loadings and unique variance matrices with EM.",
            "transform(X)": "Project X onto the latent factor subspace.",
            "fit_transform(X)": "Fit to X and return projected factors."
        }
    },
    "ica": {
        "syntax": "ml.ICA(n_components=2, max_iter=200, tol=1e-4, seed=None)",
        "description": "Independent Component Analysis using FastICA non-Gaussianity maximization",
        "category": "Dimensionality Reduction",
        "example": "ica = ml.ICA(n_components=2, seed=42)\nS_est = ica.fit_transform(X)",
        "params": {
            "n_components": "int in range [1, n_features] [default: 2] - Number of independent source signals",
            "max_iter": "int >= 1 [default: 200] - Maximum fixed-point iterations",
            "tol": "float > 0.0 [default: 1e-4] - Convergence tolerance",
            "seed": "int >= 0 or None [default: None] - Random seed for unmixing initialization"
        },
        "methods": {
            "fit(X)": "Estimate unmixing matrix using whitening and negentropy maximization.",
            "transform(X)": "Recover independent source signals.",
            "fit_transform(X)": "Fit and recover independent components from mixtures X."
        }
    },

    # Trees & Ensembles
    "tree": {
        "syntax": "ml.Tree(task='classify', max_depth=10, min_samples_split=2, max_features=None, seed=None)",
        "description": "Decision Tree (CART algorithm) for classification (Gini) or regression (Variance)",
        "category": "Trees & Ensembles",
        "example": "# Classification:\ntree = ml.Tree(task='classify', max_depth=5)\ntree.fit(X, y)\npreds = tree.predict(X)\n\n# Regression:\ntree_reg = ml.Tree(task='regress', max_depth=4)\ntree_reg.fit(X, y)",
        "params": {
            "task": "Options: 'classify' (Gini impurity) | 'regress' (Variance reduction) [default: 'classify']",
            "max_depth": "int >= 1 (e.g. 3, 5, 10) [default: 10] - Maximum depth of the decision tree",
            "min_samples_split": "int >= 2 [default: 2] - Minimum number of samples required to split an internal node",
            "max_features": "Options: None (all features) | 'sqrt' | 'log2' | int >= 1 [default: None] - Number of features to consider per split",
            "seed": "int >= 0 or None [default: None] - Random seed for feature sampling"
        },
        "methods": {
            "fit(X, y)": "Build decision tree recursively from training data.",
            "predict(X)": "Predict class or value for samples in X.",
            "predict_proba(X)": "Predict class probabilities (classification only)."
        }
    },
    "forest": {
        "syntax": "ml.Forest(n_trees=100, task='classify', max_depth=10, min_samples_split=2, max_features='sqrt', seed=None)",
        "description": "Random Forest ensemble combining bootstrap aggregation and feature subspace sampling",
        "category": "Trees & Ensembles",
        "example": "rf = ml.Forest(n_trees=50, task='classify', seed=42)\nrf.fit(X_train, y_train)\npreds = rf.predict(X_test)\nprobs = rf.predict_proba(X_test)",
        "params": {
            "n_trees": "int >= 1 (e.g. 10, 50, 100) [default: 100] - Number of trees in the forest",
            "task": "Options: 'classify' | 'regress' [default: 'classify']",
            "max_depth": "int >= 1 [default: 10] - Maximum depth of each tree",
            "min_samples_split": "int >= 2 [default: 2] - Minimum samples required to split a node",
            "max_features": "Options: 'sqrt' | 'log2' | int >= 1 | None [default: 'sqrt'] - Number of features to sample at each split",
            "seed": "int >= 0 or None [default: None] - Random seed for reproducible bagging"
        },
        "methods": {
            "fit(X, y)": "Fit ensemble of decision trees on bootstrap samples.",
            "predict(X)": "Predict class (majority voting) or mean value (regression).",
            "predict_proba(X)": "Predict averaged class probabilities."
        }
    },
    "bag": {
        "syntax": "ml.Bag(base_model=None, n_models=10, seed=None)",
        "description": "Bootstrap Aggregation (Bagging) meta-estimator for arbitrary base models",
        "category": "Trees & Ensembles",
        "example": "bag = ml.Bag(base_model=ml.Tree(task='classify'), n_models=15, seed=42)\nbag.fit(X_train, y_train)\npreds = bag.predict(X_test)",
        "params": {
            "base_model": "Any model instance implementing fit(X, y) and predict(X) (e.g. ml.Tree(), ml.Linear()) [default: Tree]",
            "n_models": "int >= 1 (e.g. 10, 20) [default: 10] - Number of estimators in the ensemble",
            "seed": "int >= 0 or None [default: None] - Random seed for bootstrap sampling"
        },
        "methods": {
            "fit(X, y)": "Fit ensemble of cloned base models on bootstrap subsets.",
            "predict(X)": "Aggregate predictions across all estimators."
        }
    },
    "adaboost": {
        "syntax": "ml.AdaBoost(n_rounds=50, lr=1.0)",
        "description": "Multi-class SAMME Adaptive Boosting with decision stump weak learners",
        "category": "Trees & Ensembles",
        "example": "ada = ml.AdaBoost(n_rounds=50, lr=0.5)\nada.fit(X_train, y_train)\npreds = ada.predict(X_test)",
        "params": {
            "n_rounds": "int >= 1 (e.g. 20, 50, 100) [default: 50] - Number of boosting iterations",
            "lr": "float in range (0.0, 1.0] [default: 1.0] - Learning rate shrinkage applied to each stump"
        },
        "methods": {
            "fit(X, y)": "Fit sequence of weighted decision stumps.",
            "predict(X)": "Predict classes using weighted stage vote."
        }
    },
    "gradientboost": {
        "syntax": "ml.GradientBoost(n_rounds=100, lr=0.1, max_depth=3, task='regress')",
        "description": "Gradient Boosting Machine fitting decision trees to negative loss gradients",
        "category": "Trees & Ensembles",
        "example": "gb = ml.GradientBoost(n_rounds=50, lr=0.1, max_depth=3, task='regress')\ngb.fit(X_train, y_train)\npreds = gb.predict(X_test)",
        "params": {
            "n_rounds": "int >= 1 (e.g. 50, 100) [default: 100] - Number of boosting stages to perform",
            "lr": "float in range (0.0, 1.0] [default: 0.1] - Shrinkage parameter reducing step contribution",
            "max_depth": "int >= 1 (e.g. 2, 3, 5) [default: 3] - Maximum depth of individual regression trees",
            "task": "Options: 'regress' (MSE residuals) | 'classify' (pseudo-residuals) [default: 'regress']"
        },
        "methods": {
            "fit(X, y)": "Fit gradient boosted trees sequentially.",
            "predict(X)": "Predict regression targets or classification labels."
        }
    },
    "xgboost": {
        "syntax": "ml.XGBoost(n_rounds=100, lr=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0, task='regress')",
        "description": "XGBoost with 2nd-order exact greedy splits (gradients + hessians) and L2 leaf regularization",
        "category": "Trees & Ensembles",
        "example": "xgb = ml.XGBoost(n_rounds=60, lr=0.1, max_depth=3, reg_lambda=1.0, task='regress')\nxgb.fit(X_train, y_train)\npreds = xgb.predict(X_test)",
        "params": {
            "n_rounds": "int >= 1 (e.g. 50, 100) [default: 100] - Number of boosting trees",
            "lr": "float in range (0.0, 1.0] [default: 0.1] - Learning rate / step shrinkage",
            "max_depth": "int >= 1 (e.g. 3, 4, 6) [default: 3] - Maximum tree depth for base learners",
            "reg_lambda": "float >= 0.0 [default: 1.0] - L2 regularization parameter on leaf weights",
            "gamma": "float >= 0.0 [default: 0.0] - Minimum loss reduction required to make a further partition",
            "task": "Options: 'regress' | 'classify' [default: 'regress']"
        },
        "methods": {
            "fit(X, y)": "Fit XGBoost ensemble using 2nd-order Taylor expansions.",
            "predict(X)": "Predict values or class labels."
        }
    },

    # Neural Networks
    "network": {
        "syntax": "ml.Network(layers)",
        "description": "Multi-Layer Perceptron container for chaining dense layers and training loops",
        "category": "Neural Networks",
        "example": "net = ml.Network([\n    ml.Dense(4, 16, activation='relu'),\n    ml.Dense(16, 2, activation='softmax')\n])\nnet.fit(X_train, y_train, optimizer=ml.Adam(lr=0.01), epochs=50, batch_size=32)\npreds = net.predict(X_test)",
        "params": {
            "layers": "list of ml.Dense layer instances forming the neural network architecture"
        },
        "methods": {
            "fit(X, y, optimizer=None, loss=None, epochs=100, batch_size=32)": "Train network using backpropagation.",
            "predict(X)": "Predict class labels or output values.",
            "predict_proba(X)": "Predict probability outputs from the final layer."
        }
    },
    "dense": {
        "syntax": "ml.Dense(in_size, out_size, activation='relu')",
        "description": "Fully connected neural network layer with Xavier / He weight initialization",
        "category": "Neural Networks",
        "example": "layer = ml.Dense(in_size=8, out_size=32, activation='relu')",
        "params": {
            "in_size": "int >= 1 - Number of input features/dimensions",
            "out_size": "int >= 1 - Number of output neurons/dimensions",
            "activation": "Options: 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'linear' [default: 'relu']"
        },
        "methods": {
            "forward(inputs)": "Compute layer forward pass: a = activation(inputs @ W + b).",
            "backward(grad_output)": "Compute gradients with respect to inputs, weights, and bias."
        }
    },
    "sgd": {
        "syntax": "ml.SGD(lr=0.01)",
        "description": "Stochastic Gradient Descent optimizer: W <- W - lr * grad",
        "category": "Neural Networks",
        "example": "opt = ml.SGD(lr=0.01)\nnet.fit(X, y, optimizer=opt, epochs=50)",
        "params": {
            "lr": "float > 0.0 [default: 0.01] - Step size learning rate"
        }
    },
    "momentum": {
        "syntax": "ml.Momentum(lr=0.01, beta=0.9)",
        "description": "SGD with Momentum optimizer accumulating velocity vector",
        "category": "Neural Networks",
        "example": "opt = ml.Momentum(lr=0.01, beta=0.9)\nnet.fit(X, y, optimizer=opt, epochs=50)",
        "params": {
            "lr": "float > 0.0 [default: 0.01] - Learning rate",
            "beta": "float in range [0.0, 1.0) [default: 0.9] - Momentum decay parameter"
        }
    },
    "rmsprop": {
        "syntax": "ml.RMSProp(lr=0.001, beta=0.999)",
        "description": "RMSProp optimizer with exponentially decaying average of squared gradients",
        "category": "Neural Networks",
        "example": "opt = ml.RMSProp(lr=0.001, beta=0.9)\nnet.fit(X, y, optimizer=opt, epochs=50)",
        "params": {
            "lr": "float > 0.0 [default: 0.001] - Learning rate",
            "beta": "float in range [0.0, 1.0) [default: 0.999] - Moving average discount factor"
        }
    },
    "adam": {
        "syntax": "ml.Adam(lr=0.001, beta1=0.9, beta2=0.999)",
        "description": "Adam optimizer combining 1st moment (momentum) and 2nd moment (RMSProp)",
        "category": "Neural Networks",
        "example": "opt = ml.Adam(lr=0.001, beta1=0.9, beta2=0.999)\nnet.fit(X, y, optimizer=opt, epochs=50)",
        "params": {
            "lr": "float > 0.0 [default: 0.001] - Learning rate",
            "beta1": "float in range [0.0, 1.0) [default: 0.9] - 1st moment exponential decay factor",
            "beta2": "float in range [0.0, 1.0) [default: 0.999] - 2nd moment exponential decay factor"
        }
    },

    # Metrics
    "mse": {
        "syntax": "ml.mse(y, pred)",
        "description": "Mean Squared Error: (1/n) * sum((y - pred)^2)",
        "category": "Metrics",
        "example": "val = ml.mse(y_true, y_pred)",
        "params": {
            "y": "1D array of ground truth target values",
            "pred": "1D array of predicted values"
        }
    },
    "rmse": {
        "syntax": "ml.rmse(y, pred)",
        "description": "Root Mean Squared Error: sqrt(MSE)",
        "category": "Metrics",
        "example": "val = ml.rmse(y_true, y_pred)",
        "params": {
            "y": "1D array of true values",
            "pred": "1D array of predicted values"
        }
    },
    "mae": {
        "syntax": "ml.mae(y, pred)",
        "description": "Mean Absolute Error: (1/n) * sum(|y - pred|)",
        "category": "Metrics",
        "example": "val = ml.mae(y_true, y_pred)",
        "params": {
            "y": "1D array of true values",
            "pred": "1D array of predicted values"
        }
    },
    "r2": {
        "syntax": "ml.r2(y, pred)",
        "description": "Coefficient of Determination R^2 score: 1 - SS_res / SS_tot",
        "category": "Metrics",
        "example": "score = ml.r2(y_true, y_pred) # 1.0 is perfect prediction",
        "params": {
            "y": "1D array of true regression targets",
            "pred": "1D array of predicted values"
        }
    },
    "accuracy": {
        "syntax": "ml.accuracy(y, pred)",
        "description": "Classification accuracy score: (correct / total)",
        "category": "Metrics",
        "example": "acc = ml.accuracy(y_true, y_pred) # e.g. 0.95 (95%)",
        "params": {
            "y": "1D array of true class labels",
            "pred": "1D array of predicted class labels"
        }
    },
    "precision": {
        "syntax": "ml.precision(y, pred)",
        "description": "Precision score: TP / (TP + FP)",
        "category": "Metrics",
        "example": "prec = ml.precision(y_true, y_pred)",
        "params": {
            "y": "1D array of true labels",
            "pred": "1D array of predicted labels"
        }
    },
    "recall": {
        "syntax": "ml.recall(y, pred)",
        "description": "Recall score (Sensitivity): TP / (TP + FN)",
        "category": "Metrics",
        "example": "rec = ml.recall(y_true, y_pred)",
        "params": {
            "y": "1D array of true labels",
            "pred": "1D array of predicted labels"
        }
    },
    "f1": {
        "syntax": "ml.f1(y, pred)",
        "description": "F1 Score: Harmonic mean of precision and recall: 2 * (P * R) / (P + R)",
        "category": "Metrics",
        "example": "f1_score = ml.f1(y_true, y_pred)",
        "params": {
            "y": "1D array of true labels",
            "pred": "1D array of predicted labels"
        }
    },
    "confusion": {
        "syntax": "ml.confusion(y, pred)",
        "description": "Compute confusion matrix showing true vs predicted class frequencies",
        "category": "Metrics",
        "example": "cm = ml.confusion(y_true, y_pred)\nprint(cm)",
        "params": {
            "y": "1D array of true labels",
            "pred": "1D array of predicted labels"
        }
    },
    "silhouette": {
        "syntax": "ml.silhouette(X, labels)",
        "description": "Compute Silhouette score in [-1, +1] measuring cluster separation and cohesion",
        "category": "Metrics",
        "example": "score = ml.silhouette(X, km.labels)",
        "params": {
            "X": "Feature matrix array of shape (n_samples, n_features)",
            "labels": "Cluster index labels of shape (n_samples,)"
        }
    },
    "wcss": {
        "syntax": "ml.wcss(X, labels, centers)",
        "description": "Within-Cluster Sum of Squares (Inertia) for evaluating clustering",
        "category": "Metrics",
        "example": "inertia = ml.wcss(X, km.labels, km.centers)",
        "params": {
            "X": "Feature matrix array",
            "labels": "Cluster assignments",
            "centers": "Centroids array"
        }
    },
    "roc_curve": {
        "syntax": "fpr, tpr, thresholds = ml.roc_curve(y_true, y_score, pos_label=1)",
        "description": "Compute Receiver Operating Characteristic (ROC) curve points for binary classification",
        "category": "Metrics",
        "example": "fpr, tpr, thresholds = ml.roc_curve(y_test, y_probs)\nscore = ml.auc(fpr, tpr)",
        "params": {
            "y_true": "1D array of true binary class labels",
            "y_score": "1D array of predicted probabilities or decision scores",
            "pos_label": "Label considered as positive class [default: 1]"
        }
    },
    "auc": {
        "syntax": "ml.auc(x, y)",
        "description": "Compute Area Under the Curve (AUC) using trapezoidal rule integration",
        "category": "Metrics",
        "example": "auc_score = ml.auc(fpr, tpr)",
        "params": {
            "x": "1D array of x coordinates (e.g. FPR)",
            "y": "1D array of y coordinates (e.g. TPR)"
        }
    },
    "roc_auc_score": {
        "syntax": "ml.roc_auc_score(y_true, y_score, pos_label=1)",
        "description": "Compute Area Under the ROC Curve (ROC AUC score) in range [0, 1]",
        "category": "Metrics",
        "example": "score = ml.roc_auc_score(y_test, model.predict_proba(x_test))",
        "params": {
            "y_true": "1D array of true labels",
            "y_score": "1D array of predicted probabilities or confidence scores",
            "pos_label": "Label of positive class [default: 1]"
        }
    },

    # Validation & Hyperparameter Tuning
    "cross_validate": {
        "syntax": "ml.cross_validate(model, X, y, cv=5, metric=None, seed=None)",
        "description": "Evaluate estimator performance using K-Fold, Stratified, or custom cross-validation",
        "category": "Validation",
        "example": "scores = ml.cross_validate(model, X, y, cv=5, seed=42)\nprint('Mean Score:', scores.mean(), 'Std:', scores.std())",
        "params": {
            "model": "Model instance supporting .fit(X, y) and .predict(X)",
            "X": "Feature matrix",
            "y": "Target vector",
            "cv": "int (e.g. 5, 10) or splitter instance (ml.KFold, ml.StratifiedKFold, ml.LeaveOneOut, ml.ShuffleSplit, ml.TimeSeriesSplit) [default: 5]",
            "metric": "Callable scoring function metric(y_true, y_pred) [default: None - auto accuracy/r2]",
            "seed": "int >= 0 or None [default: None] - Random seed for fold partitioning"
        }
    },
    "kfold": {
        "syntax": "ml.KFold(n_splits=5, shuffle=True, seed=None)",
        "description": "K-Fold cross-validator splitting dataset into k consecutive or shuffled folds",
        "category": "Validation",
        "example": "kf = ml.KFold(n_splits=5, shuffle=True, seed=42)\nscores = ml.cross_validate(model, X, y, cv=kf)",
        "params": {
            "n_splits": "int >= 2 (e.g. 3, 5, 10) [default: 5] - Number of folds",
            "shuffle": "bool True or False [default: True] - Whether to shuffle before splitting",
            "seed": "int >= 0 or None [default: None] - Random seed"
        }
    },
    "stratifiedkfold": {
        "syntax": "ml.StratifiedKFold(n_splits=5, shuffle=True, seed=None)",
        "description": "Stratified K-Fold cross-validator ensuring balanced class distribution per fold",
        "category": "Validation",
        "example": "skf = ml.StratifiedKFold(n_splits=5, shuffle=True, seed=42)\nscores = ml.cross_validate(classifier, X, y, cv=skf)",
        "params": {
            "n_splits": "int >= 2 [default: 5] - Number of folds",
            "shuffle": "bool True or False [default: True] - Shuffle data before splitting",
            "seed": "int >= 0 or None [default: None] - Random seed"
        }
    },
    "leaveoneout": {
        "syntax": "ml.LeaveOneOut() (alias: ml.LOOCV())",
        "description": "Leave-One-Out cross-validator where each sample serves as a test fold of size 1",
        "category": "Validation",
        "example": "loo = ml.LeaveOneOut()\nscores = ml.cross_validate(model, X, y, cv=loo)",
        "params": {}
    },
    "shufflesplit": {
        "syntax": "ml.ShuffleSplit(n_splits=5, test_size=0.2, seed=None)",
        "description": "Random permutation cross-validator generating repeated randomized train/test splits",
        "category": "Validation",
        "example": "ss = ml.ShuffleSplit(n_splits=10, test_size=0.25, seed=42)\nscores = ml.cross_validate(model, X, y, cv=ss)",
        "params": {
            "n_splits": "int >= 1 [default: 5] - Number of re-shuffling iterations",
            "test_size": "float in (0, 1) [default: 0.2] - Proportion of dataset to include in test split",
            "seed": "int or None [default: None] - Random seed"
        }
    },
    "timeseriessplit": {
        "syntax": "ml.TimeSeriesSplit(n_splits=5)",
        "description": "Time Series cross-validator using forward-chaining splits without lookahead data leakage",
        "category": "Validation",
        "example": "tss = ml.TimeSeriesSplit(n_splits=5)\nscores = ml.cross_validate(model, X, y, cv=tss)",
        "params": {
            "n_splits": "int >= 2 [default: 5] - Number of forward splits"
        }
    },
    "gridsearchcv": {
        "syntax": "ml.GridSearchCV(model, param_grid, cv=5, metric=None, seed=None)",
        "description": "Exhaustive hyperparameter grid search cross-validation optimizer",
        "category": "Validation",
        "example": "grid = ml.GridSearchCV(\n    ml.SVM(),\n    param_grid={'C': [0.1, 1.0, 10.0], 'gamma': [0.1, 1.0]},\n    cv=5\n)\ngrid.fit(X_train, y_train)\ngrid.summary()\npreds = grid.predict(X_test)",
        "params": {
            "model": "Base estimator model instance (e.g. ml.SVM(), ml.Linear(), ml.Forest())",
            "param_grid": "Dictionary of lists mapping parameter names to lists of candidate values",
            "cv": "int (number of folds) or CV splitter instance [default: 5]",
            "metric": "Callable scoring function or None [default: None - auto accuracy/r2]",
            "seed": "int or None [default: None] - Random seed"
        },
        "methods": {
            "fit(X, y)": "Fit all hyperparameter combinations and retrain best model on full dataset",
            "predict(X)": "Predict target values using the best tuned model",
            "predict_proba(X)": "Predict class probabilities using best tuned model",
            "summary()": "Print ranked table of all explored parameters and CV scores"
        }
    },
    "randomsearchcv": {
        "syntax": "ml.RandomSearchCV(model, param_distributions, n_iter=10, cv=5, metric=None, seed=None)",
        "description": "Randomized hyperparameter search cross-validation optimizer",
        "category": "Validation",
        "example": "search = ml.RandomSearchCV(\n    ml.Forest(),\n    param_distributions={'n_trees': [20, 50, 100], 'max_depth': [3, 5, 10]},\n    n_iter=6,\n    cv=3\n)\nsearch.fit(X_train, y_train)\nsearch.summary()",
        "params": {
            "model": "Base estimator template instance",
            "param_distributions": "Dictionary mapping parameter names to candidate lists",
            "n_iter": "int >= 1 [default: 10] - Number of randomized parameter combinations to test",
            "cv": "int or CV splitter [default: 5]",
            "metric": "Callable scoring function or None",
            "seed": "int or None - Random seed"
        },
        "methods": {
            "fit(X, y)": "Evaluate random combinations and fit best model",
            "predict(X)": "Predict using best model",
            "summary()": "Print ranked summary of tested configurations"
        }
    },

    # Plotting
    "scatter": {
        "syntax": "ml.scatter(x, y, c=None, color='royalblue', label=None, title='Scatter Plot', xlabel='X', ylabel='Y', show=True)",
        "description": "Scatter plot with custom colors, legend labels, and colormap support",
        "category": "Plotting",
        "example": "# Standalone scatter:\nml.scatter(X, y, color='royalblue', label='Samples')\n\n# Overlay scatter with fitted line on same plot:\nml.scatter(X, y, color='blue', label='Data', show=False)\nml.line(x_test, y_pred, color='red', label='Fit', equation=model, show=True)",
        "params": {
            "x": "1D or 2D array / Series of x coordinates",
            "y": "1D or 2D array / Series of y coordinates",
            "c": "Optional label or numeric array for continuous colormap coloring [default: None]",
            "color": "Options: 'royalblue' | 'red' | 'blue' | 'green' | 'black' | 'purple' | 'teal' | 'crimson' | hex (e.g. '#2ca02c') [default: 'royalblue']",
            "label": "Optional legend label string (e.g. 'Data Points', 'Test Set') [default: None]",
            "title": "str [default: 'Scatter Plot'] - Plot title",
            "xlabel": "str [default: 'X'] - Label for X axis",
            "ylabel": "str [default: 'Y'] - Label for Y axis",
            "show": "bool True or False [default: True] - If False, keeps figure open to overlay further lines/scatters before showing"
        }
    },
    "hist": {
        "syntax": "ml.hist(data, col=None, bins=20, color='skyblue', title=None, xlabel=None, ylabel='Frequency', show=True)",
        "description": "Plot histogram distribution of a DataFrame column or 1D array",
        "category": "Plotting",
        "example": "ml.hist(df, 'age', bins=25, color='teal')",
        "params": {
            "data": "pandas DataFrame or 1D array of values",
            "col": "str or None - Name of the numerical column if data is DataFrame",
            "bins": "int >= 1 (e.g. 10, 20, 50) [default: 20] - Number of histogram bins",
            "color": "Options: 'skyblue' | 'navy' | 'crimson' | 'green' | 'orange' | 'purple' | hex [default: 'skyblue']",
            "title": "Optional plot title string [default: auto]",
            "xlabel": "Optional X-axis label string [default: column name]",
            "ylabel": "str [default: 'Frequency']",
            "show": "bool True or False [default: True]"
        }
    },
    "line": {
        "syntax": "ml.line(x, y, color='crimson', label=None, equation=None, title='Line Plot', xlabel='X', ylabel='Y', show=True)",
        "description": "Plot 2D line with color selection, legend label, and optional equation printing / annotation",
        "category": "Plotting",
        "example": "# Line with fitted equation from model:\nml.line(x_test, y_pred, color='crimson', label='Prediction', equation=model)\n\n# Line with custom equation string:\nml.line(x_vals, y_vals, color='navy', equation='y = 2.5x + 1.0')",
        "params": {
            "x": "1D array of x coordinates (automatically sorted for clean connected lines)",
            "y": "1D array of y coordinates",
            "color": "Options: 'crimson' | 'blue' | 'green' | 'red' | 'black' | 'purple' | 'orange' | 'teal' | hex (e.g. '#FF5733') [default: 'crimson']",
            "label": "Optional legend label string (e.g. 'Fitted Line', 'Prediction') [default: None]",
            "equation": "Options: None | True (auto-fit line equation) | model instance (e.g. ml.Linear instance) | str (e.g. 'y = 2.5x + 1.0') - Automatically prints and annotates equation [default: None]",
            "title": "str [default: 'Line Plot'] - Plot title",
            "xlabel": "str [default: 'X'] - Label for X axis",
            "ylabel": "str [default: 'Y'] - Label for Y axis",
            "show": "bool True or False [default: True] - If False, keeps figure open to add more plots before showing"
        }
    },
    "loss_plot": {
        "syntax": "ml.loss_plot(losses, color='crimson', title='Training Loss Curve', xlabel='Iteration / Epoch', ylabel='Loss', show=True)",
        "description": "Plot training loss / cost convergence curve across iterations or epochs",
        "category": "Plotting",
        "example": "ml.loss_plot(model, color='purple')",
        "params": {
            "losses": "Model instance (e.g. model) or list/array of recorded losses (e.g. model.losses)",
            "color": "Options: 'crimson' | 'blue' | 'purple' | 'green' | 'black' | hex [default: 'crimson']",
            "title": "str [default: 'Training Loss Curve']",
            "xlabel": "str [default: 'Iteration / Epoch']",
            "ylabel": "str [default: 'Loss']",
            "show": "bool True or False [default: True]"
        }
    },
    "boundary": {
        "syntax": "ml.boundary(model, X, y, title='Decision Boundary', resolution=200, cmap='coolwarm', show=True)",
        "description": "Plot 2D classifier decision boundary contour mesh with data points overlay",
        "category": "Plotting",
        "example": "ml.boundary(model, X[:, :2], y, cmap='coolwarm')",
        "params": {
            "model": "Fitted classifier instance with .predict(X) method",
            "X": "2D feature matrix of shape (n_samples, 2)",
            "y": "Class labels of shape (n_samples,)",
            "title": "str [default: 'Decision Boundary']",
            "resolution": "int >= 50 (e.g. 100, 200, 300) [default: 200] - Mesh grid point density",
            "cmap": "Colormap options: 'coolwarm' | 'viridis' | 'plasma' | 'bwr' | 'Set1' [default: 'coolwarm']",
            "show": "bool True or False [default: True]"
        }
    },
    "confusion_plot": {
        "syntax": "ml.confusion_plot(y, pred, cmap='Blues', title='Confusion Matrix', show=True)",
        "description": "Plot confusion matrix heatmap with count annotations",
        "category": "Plotting",
        "example": "ml.confusion_plot(y_test, preds, cmap='Blues')",
        "params": {
            "y": "1D array of true class labels",
            "pred": "1D array of predicted class labels",
            "cmap": "Colormap options: 'Blues' | 'Greens' | 'Purples' | 'Oranges' | 'YlOrRd' [default: 'Blues']",
            "title": "str [default: 'Confusion Matrix']",
            "show": "bool True or False [default: True]"
        }
    },
    "roc_plot": {
        "syntax": "ml.roc_plot(y_true, y_score, pos_label=1, color='crimson', title='ROC Curve', show=True)",
        "description": "Plot Receiver Operating Characteristic (ROC) curve with AUC annotation and random diagonal baseline",
        "category": "Plotting",
        "example": "ml.roc_plot(y_test, model.predict_proba(x_test)[:, 1], color='crimson')",
        "params": {
            "y_true": "1D array of true binary class labels",
            "y_score": "1D array of predicted probabilities or decision function values",
            "pos_label": "Positive class label [default: 1]",
            "color": "Curve line color [default: 'crimson']",
            "title": "str [default: 'Receiver Operating Characteristic (ROC)']",
            "show": "bool True or False [default: True]"
        }
    },
    "residual_plot": {
        "syntax": "ml.residual_plot(y_true, y_pred, color='royalblue', title='Residual Plot', show=True)",
        "description": "Plot regression residuals (y_true - y_pred) vs fitted predictions with zero-error line",
        "category": "Plotting",
        "example": "ml.residual_plot(y_test, y_pred, color='royalblue')",
        "params": {
            "y_true": "1D array of actual target values",
            "y_pred": "1D array of predicted values from regression model",
            "color": "Scatter point color [default: 'royalblue']",
            "title": "str [default: 'Residual Plot']",
            "show": "bool True or False [default: True]"
        }
    },
    "feature_importance": {
        "syntax": "ml.feature_importance(model, feature_names=None, top_n=10, color='royalblue', title='Feature Importance', show=True)",
        "description": "Plot ranked horizontal bar chart of relative feature importances for Trees, Forests, or Linear models",
        "category": "Plotting",
        "example": "ml.feature_importance(forest_model, feature_names=['Age', 'Income', 'Score'], top_n=5)",
        "params": {
            "model": "Fitted model with feature_importances_ or weights (e.g. ml.Tree, ml.Forest, ml.Linear)",
            "feature_names": "list of str feature names [default: None -> Feature 1, Feature 2...]",
            "top_n": "int >= 1 [default: 10] - Number of top features to display",
            "color": "Bar color [default: 'royalblue']",
            "title": "str [default: 'Feature Importance']",
            "show": "bool True or False [default: True]"
        }
    },
    "dendrogram": {
        "syntax": "ml.dendrogram(model, color='purple', title='Hierarchical Dendrogram', show=True)",
        "description": "Plot hierarchical clustering dendrogram tree",
        "category": "Plotting",
        "example": "ml.dendrogram(hc_model, color='purple')",
        "params": {
            "model": "Fitted ml.Hierarchical instance",
            "color": "Options: 'purple' | 'navy' | 'darkgreen' | 'crimson' | hex [default: 'purple']",
            "title": "str [default: 'Hierarchical Dendrogram']",
            "show": "bool True or False [default: True]"
        }
    },
    "pca_plot": {
        "syntax": "ml.pca_plot(X, y=None, color='teal', cmap='tab10', title='PCA Projection', show=True)",
        "description": "Compute and visualize 2D PCA projection scatter plot",
        "category": "Plotting",
        "example": "ml.pca_plot(X, y=y_labels, cmap='tab10')",
        "params": {
            "X": "High-dimensional feature matrix",
            "y": "Optional label array for color-coding points [default: None]",
            "color": "Color if y is None: 'teal' | 'royalblue' | 'purple' [default: 'teal']",
            "cmap": "Colormap when y is supplied: 'tab10' | 'viridis' | 'rainbow' [default: 'tab10']",
            "title": "str [default: 'PCA Projection']",
            "show": "bool True or False [default: True]"
        }
    },
    "cluster_plot": {
        "syntax": "ml.cluster_plot(X, labels, centers=None, cmap='rainbow', title='Cluster Assignments', show=True)",
        "description": "Visualize 2D cluster groupings with distinct cluster colors and centroid markers",
        "category": "Plotting",
        "example": "ml.cluster_plot(X, km.labels, centers=km.centers)",
        "params": {
            "X": "2D feature matrix",
            "labels": "Cluster index assignments",
            "centers": "Optional cluster centroid coordinates to mark with 'X' [default: None]",
            "cmap": "Colormap options: 'rainbow' | 'viridis' | 'tab10' | 'Set2' [default: 'rainbow']",
            "title": "str [default: 'Cluster Assignments']",
            "show": "bool True or False [default: True]"
        }
    },

    # Help
    "help": {
        "syntax": "ml.help(name=None)",
        "description": "Show interactive help, allowed argument values/ranges, and raw code examples",
        "category": "Help",
        "example": "ml.help()          # overview of all modules with examples\nml.help('linear')  # detailed parameters, options, and constraints\nml.help(\"line\")    # works in single or double quotes",
        "params": {
            "name": "str or None [default: None] - Specific function, model, or category name to inspect (supports single and double quotes)"
        }
    }
}


def help(name=None):
    """Show help for all functions or a specific function."""
    if name is None:
        categories = {}
        for key, info in _HELP.items():
            if info.get("group"):
                continue
            cat = info.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            
            categories[cat].append((key, info["syntax"], info["description"], info.get("example", "")))
            
        print("=" * 80)
        print(" mymllib Complete Function Catalog & Raw Usage Examples ".center(80, "="))
        print("=" * 80)
        print("\nTip: Use ml.help('name') or CLI: mymllib \"help(name)\" to see allowed argument values & ranges.\n")
        
        order = [
            "Preprocessing", "Supervised Models", "Unsupervised Models", 
            "Dimensionality Reduction", "Trees & Ensembles", "Neural Networks", 
            "Metrics", "Validation", "Plotting", "Help"
        ]
        
        for cat in order:
            if cat in categories:
                print(f"+-- [ {cat} ] " + "-" * (73 - len(cat)))
                for k, syntax, desc, ex in categories[cat]:
                    print(f"| * {k.upper()}")
                    print(f"|   Syntax  : {syntax}")
                    print(f"|   Details : {desc}")
                    if ex:
                        ex_lines = ex.strip().split('\n')
                        print(f"|   Example : {ex_lines[0]}")
                        for line in ex_lines[1:]:
                            print(f"|             {line}")
                    print("|")
                print("+" + "-" * 78 + "\n")
                
    else:
        # Strip all single and double quotes, whitespace, and prefixes
        cleaned = str(name).strip("'\" \t\r\n")
        if cleaned.lower().startswith("ml."):
            cleaned = cleaned[3:].strip("'\" \t\r\n")
        if cleaned.lower().startswith("help(") and cleaned.endswith(")"):
            cleaned = cleaned[5:-1].strip("'\" \t\r\n")

        name_lower = cleaned.lower()

        # Convenient aliases
        aliases = {
            "standardscaler": "scaler",
            "minmaxscaler": "normalizer",
            "onehotencoder": "encoder",
            "logisticregression": "logistic",
            "linearregression": "linear",
            "randomforest": "forest",
            "decisiontree": "tree",
            "gradientboosting": "gradientboost",
            "mlp": "network",
            "loocv": "leaveoneout",
            "gridsearch": "gridsearchcv",
            "randomsearch": "randomsearchcv",
            "roc": "roc_plot",
            "roc_auc": "roc_auc_score",
            "featureimportance": "feature_importance",
            "residuals": "residual_plot",
        }
        if name_lower in aliases:
            name_lower = aliases[name_lower]

        if name_lower not in _HELP:
            print(f'No help found for "{name}". Use ml.help() to see all available functions.')
            return
            
        info = _HELP[name_lower]
        
        if info.get("group"):
            print("=" * 70)
            print(f" Group: {name_lower.upper()} ".center(70, "="))
            print("=" * 70)
            for var in info.get("variants", []):
                var_info = _HELP.get(var.lower())
                if var_info:
                    print(f"\n[Variant: {var}]")
                    print(f"Syntax      : {var_info['syntax']}")
                    print(f"Description : {var_info['description']}")
                    if var_info.get("example"):
                        print(f"Example     :\n{var_info['example']}")
            print("=" * 70 + "\n")
        else:
            print("=" * 70)
            print(f" Help: {name_lower.upper()} ({info['category']}) ".center(70, "="))
            print("=" * 70)
            print(f"Syntax      : {info['syntax']}")
            print(f"Description : {info['description']}")
            print(f"Category    : {info['category']}")
            
            # Display detailed parameter options and domains
            if info.get("params"):
                print("\nArguments & Allowed Values / Domains:")
                for param_name, param_desc in info["params"].items():
                    print(f"  * {param_name:<12} : {param_desc}")
            else:
                print("\nArguments   : None (Instantiate directly)")
                
            # Display methods if class
            if info.get("methods"):
                print("\nAvailable Methods:")
                for method_sig, method_desc in info["methods"].items():
                    print(f"  * {method_sig:<28} : {method_desc}")
                    
            # Display concrete code example
            if info.get("example"):
                print("\nRaw Python Code Example:")
                print("------------------------------------------------------------")
                print(info["example"])
                print("------------------------------------------------------------")
                
            if "variants" in info:
                print(f"\nRelated Variants : {', '.join(info['variants'])}")
            print("=" * 70 + "\n")
