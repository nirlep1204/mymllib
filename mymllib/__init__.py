# mymllib — A personal machine learning library built from scratch.

# Preprocessing & Data Utilities
from .data.preprocess import Scaler, Normalizer, Encoder, LabelEncoder, split, xy

# Supervised Models
from .models.linear import Linear
from .models.logistic import Logistic
from .models.softmax import Softmax
from .models.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from .models.gda import GDA
from .models.knn import KNN
from .models.svm import SVM
from .models.svr import SVR
from .models.glm import GLM
from .models.perceptron import Perceptron

# Clustering
from .cluster.kmeans import KMeans
from .cluster.hierarchical import Hierarchical
from .cluster.gmm import GMM

# Dimensionality Reduction
from .reduce.pca import PCA
from .reduce.factor import Factor
from .reduce.ica import ICA

# Trees & Ensembles
from .trees.tree import Tree
from .trees.forest import Forest
from .trees.bag import Bag
from .trees.adaboost import AdaBoost
from .trees.gradient_boost import GradientBoost
from .trees.xgboost import XGBoost

# Neural Networks
from .nn.layers import Dense
from .nn.optimizers import SGD, Momentum, RMSProp, Adam
from .nn.network import Network

# Metrics
from .metrics.scores import (
    accuracy,
    auc,
    confusion,
    f1,
    mae,
    mse,
    precision,
    r2,
    recall,
    rmse,
    roc_auc_score,
    roc_curve,
    silhouette,
    wcss,
)

# Validation & Hyperparameter Tuning
from .validation.cv import (
    KFold,
    LeaveOneOut,
    LOOCV,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from .validation.search import GridSearchCV, RandomSearchCV

# Plotting & Visualizations
from .plot.plots import (
    boundary,
    cluster_plot,
    confusion_plot,
    dendrogram,
    feature_importance,
    hist,
    line,
    loss_plot,
    pca_plot,
    residual_plot,
    roc_plot,
    scatter,
)

# Interactive Help
from .help import help
