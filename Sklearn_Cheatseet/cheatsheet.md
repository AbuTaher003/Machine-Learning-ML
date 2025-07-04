# 🎯 Scikit-learn (Sklearn) Cheatsheet — Code + Bangla Explanation

---

## ১. Data Generation (ডেটা তৈরি)

```python
from sklearn.datasets import make_classification, make_regression, make_blobs

# ক্লাসিফিকেশন ডেটা তৈরি করা
X_cls, y_cls = make_classification(
    n_samples=100,           # মোট ১০০টা ডেটা পয়েন্ট তৈরি করবে
    n_features=2,            # প্রতিটা পয়েন্টে ২টা ফিচার থাকবে
    n_informative=1,         # ১টা ফিচার আসলেই ক্লাস নির্ধারণে গুরুত্বপূর্ণ
    n_redundant=0,           # কোনো redundant ফিচার থাকবে না (অপ্রয়োজনীয় কপি)
    n_classes=2,             # ২টা আলাদা ক্লাস (যেমন: ০ এবং ১)
    n_clusters_per_class=1,  # প্রতিটি ক্লাসে ১টা cluster থাকবে (group)
    class_sep=2.0,           # দুই ক্লাসের ডেটার মাঝে দূরত্ব কত থাকবে (বড় হলে আলাদা করা সহজ)
    random_state=42          # র‍্যান্ডম সিড ধরে রাখা, যাতে একই ডেটা বার বার পাওয়া যায়
)

# রিগ্রেশন ডেটা তৈরি করা
X_reg, y_reg = make_regression(
    n_samples=100,           # মোট ১০০টা ডেটা পয়েন্ট
    n_features=1,            # ১টা ফিচার থাকবে
    noise=10,                # আউটপুটে কিছু noise বা randomness যোগ করা হবে, বাস্তবমুখী করতে
    random_state=42          # রিপ্রোডিউসিবিলিটি জন্য সিড ধরে রাখা
)

# ক্লাস্টারিং ডেটা তৈরি করা
X_clust, y_clust = make_blobs(
    n_samples=100,           # মোট ১০০টা ডেটা পয়েন্ট
    centers=3,               # ৩টা আলাদা ক্লাস্টার বা গ্রুপ তৈরি করবে
    n_features=2,            # ২টা ফিচার থাকবে প্রতিটাতে
    cluster_std=1.0,         # প্রতিটি ক্লাস্টারের ভিতরে ডেটার spread বা বিস্তার
    random_state=42          # রিপ্রোডিউসিবিলিটি জন্য সিড ধরে রাখা
)

```

> **ব্যাখ্যা:**
>
> * `make_classification`: supervised ক্লাসিফিকেশন ডেটা বানায়।
> * `make_regression`: supervised রিগ্রেশন ডেটা বানায়।
> * `make_blobs`: unsupervised ক্লাস্টারিং ডেটা বানায়।

---

## ২. Data Splitting (ডেটা ভাগ করা)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
```

> **ব্যাখ্যা:**
> ডেটাকে ট্রেনিং (৮০%) এবং টেস্ট (২০%) এ ভাগ করে, মডেল শেখানোর জন্য।

---

## ৩. Preprocessing (ডেটা প্রস্তুতি)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# স্কেলিং (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# লেবেল এনকোডিং (categorical to number)
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# One Hot Encoding (categorical to binary columns)
encoder = OneHotEncoder(sparse=False)
X_onehot = encoder.fit_transform(X_train.reshape(-1, 1))

# মিসিং ডেটা পূরণ
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_train)

# পলিনোমিয়াল ফিচার তৈরি
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
```

> **ব্যাখ্যা:**
> ডেটাকে মডেলের জন্য উপযোগী করে তোলা হয়। স্কেলিং, এনকোডিং, মিসিং ভ্যালু পূরণ ইত্যাদি করে।

---

## ৪. মডেল তৈরি ও শেখানো

### Classification (ক্লাসিফিকেশন মডেল)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Regression (রিগ্রেশন মডেল)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## ৫. Model Evaluation (মডেল যাচাই)

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## ৬. Cross-validation ও Hyperparameter Tuning

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)

# Grid Search for hyperparameter tuning
params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

---

## ৭. Pipelines (প্রসেস একত্রিত করা)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

## ৮. Feature Selection (ফিচার নির্বাচন)

```python
from sklearn.feature_selection import SelectKBest, chi2, RFE

selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X_train, y_train)

# Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train, y_train)
```

---

## ৯. Clustering (ক্লাস্টারিং)

```python
from sklearn.cluster import KMeans, DBSCAN

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
labels = kmeans.labels_

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_train)
labels_db = dbscan.labels_
```

---

## ১০. Dimensionality Reduction (মাত্রা হ্রাস)

```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_train)
```

---

## ১১. Text Feature Extraction (টেক্সট থেকে ফিচার)

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = ["machine learning is fun", "sklearn is powerful"]

cv = CountVectorizer()
X_counts = cv.fit_transform(texts)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(texts)
```

---

## ১২. Model Saving and Loading (মডেল সংরক্ষণ ও পুনরায় ব্যবহার)

```python
import joblib

joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

## ১৩. Utility Functions (সহজ কাজের ফাংশন)

```python
model.get_params()      # মডেলের বর্তমান সেটিংস দেখো
model.set_params()      # নতুন সেটিংস দাও
model.score(X_test, y_test)  # Accuracy বা R2 স্কোর পাও
```

---

