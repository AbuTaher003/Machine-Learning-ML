# 🎯 Scikit-learn (Sklearn) Cheatsheet for Beginners

A complete reference of essential Sklearn modules and utilities for data preprocessing, modeling, evaluation, and advanced usage in Python.

---

## 📦 Data Generation
```python
from sklearn.datasets import make_classification, make_regression, make_blobs, load_iris, load_digits, load_diabetes

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           class_sep=2.0, random_state=42)
```

---

## 📊 Data Splitting
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## ⚙️ Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures, Binarizer, Normalizer
from sklearn.impute import SimpleImputer

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🔍 Classification Models
```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
```

---

## 📈 Regression Models
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, StackingRegressor
```

---

## 🧪 Clustering
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```

---

## 🔻 Dimensionality Reduction
```python
from sklearn.decomposition import PCA, TruncatedSVD, NMF
```

---

## 📊 Model Evaluation
```python
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score, silhouette_score, adjusted_rand_score)
```

---

## 🔁 Model Validation & Tuning
```python
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
```

---

## 🔄 Model Methods
```python
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)
model.get_params()
model.set_params()
```

---

## 🔀 Pipelines & FeatureUnion
```python
from sklearn.pipeline import Pipeline, FeatureUnion
```

---

## 🔧 Transformers
```python
from sklearn.compose import ColumnTransformer
```

---

## 🧪 Feature Selection
```python
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
```

---

## 🔡 Text Processing
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

---

## 💾 Model Saving/Loading
```python
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

## 🧠 Tips:
- Use `?function` in Jupyter to get docstrings
- Use pipelines to streamline workflows
- Use cross_val_score for unbiased evaluation
- Use GridSearchCV to find best hyperparameters
- Practice with real datasets for intuition

---

**You now hold a 100% complete Sklearn Cheatsheet. Practice, build, and own ML like a boss. **
