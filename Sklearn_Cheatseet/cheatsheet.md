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

scaler = StandardScaler()                      # StandardScaler দিয়ে ডেটা স্কেলিং করা (Mean=0, Std=1)
X_scaled = scaler.fit_transform(X_train)      # fit_transform করে ডেটার mean/std বের করে স্কেলিং করে

le = LabelEncoder()                            # LabelEncoder দিয়ে ক্যাটেগরিকাল লেবেলকে নম্বরে রূপান্তর করা
y_encoded = le.fit_transform(y_train)         # fit_transform দিয়ে লেবেলকে unique নম্বর দেয়

encoder = OneHotEncoder(sparse=False)          # OneHotEncoder দিয়ে ক্যাটেগরিকাল ফিচারকে বাইনারি কলামে রূপান্তর করা
X_onehot = encoder.fit_transform(X_train.reshape(-1, 1))  # reshape(-1,1) দিয়ে 1D কে 2D বানানো হয়

imputer = SimpleImputer(strategy='mean')      # SimpleImputer দিয়ে মিসিং ভ্যালু (NaN) পূরণ করা
X_imputed = imputer.fit_transform(X_train)    # মিসিং জায়গায় কলামের গড় মান বসানো হয়

poly = PolynomialFeatures(degree=2)           # PolynomialFeatures দিয়ে degree=2 পর্যন্ত নতুন ফিচার তৈরি
X_poly = poly.fit_transform(X_train)           # যেমন: x², x*y ইত্যাদি interaction terms তৈরি হয়
```

> **ব্যাখ্যা:**
> ডেটাকে মডেলের জন্য উপযোগী করে তোলা হয়। স্কেলিং, এনকোডিং, মিসিং ভ্যালু পূরণ ইত্যাদি করে।

---

## ৪. মডেল তৈরি ও শেখানো

### Classification (ক্লাসিফিকেশন মডেল)

```python
from sklearn.linear_model import LogisticRegression          # Logistic Regression মডেল
from sklearn.tree import DecisionTreeClassifier               # Decision Tree Classifier মডেল
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble মডেল
from sklearn.svm import SVC                                   # Support Vector Classifier (SVM)

model = LogisticRegression()                                 # Logistic Regression মডেল ইনস্ট্যান্স তৈরি
model.fit(X_train, y_train)                                  # ট্রেনিং ডেটা দিয়ে মডেল শেখানো
y_pred = model.predict(X_test)                               # টেস্ট ডেটা দিয়ে প্রিডিকশন করা

```

### Regression (রিগ্রেশন মডেল)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso                 # Linear, Ridge, Lasso regression মডেল
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor   # Ensemble regression মডেল

model = LinearRegression()                               # Linear Regression মডেল ইনস্ট্যান্স তৈরি
model.fit(X_train, y_train)                              # ট্রেনিং ডেটা দিয়ে মডেল শেখানো
y_pred = model.predict(X_test)                           # টেস্ট ডেটা দিয়ে প্রিডিকশন করা

```

---

## ৫. Model Evaluation (মডেল যাচাই)

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score

print("Accuracy:", accuracy_score(y_test, y_pred))                     # সঠিক ভবিষ্যতের হার (Accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))        # মডেলের ভুল-সঠিক ভবিষ্যতের ম্যাট্রিক্স
print("Classification Report:\n", classification_report(y_test, y_pred))  # Precision, Recall, F1-score এর বিস্তারিত রিপোর্ট
print("F1 Score:", f1_score(y_test, y_pred))                           # Precision ও Recall এর হরমোনিক মীন
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))                 # ROC curve এর নিচের এলাকা, মডেলের পারফরম্যান্স মাপার মানদণ্ড

```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))      # Mean Squared Error (গড় বর্গমূল ত্রুটি), ত্রুটির পরিমাণ মাপার জন্য
print("MAE:", mean_absolute_error(y_test, y_pred))     # Mean Absolute Error (গড় সঠিক ত্রুটি), ত্রুটির গড় মাপ
print("R2 Score:", r2_score(y_test, y_pred))            # R² স্কোর, মডেলের ব্যাখ্যা ক্ষমতা (১=পারফেক্ট, ০=খারাপ)

```

---

## ৬. Cross-validation ও Hyperparameter Tuning

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross validation দিয়ে মডেলের পারফরম্যান্স যাচাই করা
scores = cross_val_score(model, X_train, y_train, cv=5)    # ৫ ভাগে ডেটা ভাগ করে বারবার মডেল ট্রেন ও টেস্ট করে
print("Cross-validation scores:", scores)                 # প্রতিবারের স্কোর দেখাবে

# Grid Search দিয়ে Hyperparameter Tuning করা
params = {'C': [0.1, 1, 10]}                              # Logistic Regression এর C প্যারামিটার ভ্যালু সেট করা
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)  # ৫ ফোল্ড ক্রস-ভ্যালিডেশনে বেস্ট প্যারামিটার খোঁজা
grid.fit(X_train, y_train)                                 # মডেল ট্রেন করা
print("Best params:", grid.best_params_)                  # সেরা প্যারামিটার দেখাবে

```

---

## ৭. Pipelines (প্রসেস একত্রিত করা)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),           # প্রথমে ডেটা স্কেলিং করবে (Mean=0, Std=1)
    ('model', LogisticRegression())         # তারপর Logistic Regression মডেল ট্রেন করবে
])

pipe.fit(X_train, y_train)                   # Pipeline পুরো প্রোসেস একসাথে ট্রেনিং
y_pred = pipe.predict(X_test)                # টেস্ট ডেটায় প্রিডিকশন করবে

```

---

## ৮. Feature Selection (ফিচার নির্বাচন)

```python
from sklearn.feature_selection import SelectKBest, chi2, RFE

selector = SelectKBest(score_func=chi2, k=5)      # Chi-square স্কোর দিয়ে সেরা ৫টা ফিচার বেছে নেওয়া
X_new = selector.fit_transform(X_train, y_train) # ফিচার সিলেকশনের পর নতুন ডেটাসেট

# Recursive Feature Elimination (RFE) - পুনরাবৃত্তিমূলক ফিচার বাদ দেয়া
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()                       # বেস মডেল হিসেবে Logistic Regression
rfe = RFE(model, n_features_to_select=5)          # ৫টা ফিচার বাছাই করার জন্য RFE অবজেক্ট
X_rfe = rfe.fit_transform(X_train, y_train)       # ফিচার নির্বাচন করে নতুন ডেটাসেট তৈরি

```

---

## ৯. Clustering (ক্লাস্টারিং)

```python
from sklearn.cluster import KMeans, DBSCAN

kmeans = KMeans(n_clusters=3)         # KMeans ক্লাস্টারিং, ৩টা গ্রুপ বানাবে
kmeans.fit(X_train)                   # ডেটা দিয়ে ক্লাস্টারিং শেখানো
labels = kmeans.labels_               # প্রতিটি ডেটার ক্লাস্টার লেবেল (0,1,2)

dbscan = DBSCAN(eps=0.5, min_samples=5)  # DBSCAN ক্লাস্টারিং, eps=0.5 (দূরত্ব), min_samples=5 (নূন্যতম পয়েন্ট)
dbscan.fit(X_train)                      # ডেটা দিয়ে ক্লাস্টারিং শেখানো
labels_db = dbscan.labels_               # ডেটার ক্লাস্টার লেবেল (-1 মানে noise/outlier)

```

---

## ১০. Dimensionality Reduction (মাত্রা হ্রাস)

```python
ঠিক ভাই, ডান পাশে কমেন্ট সহ কোডটা দিলাম:

```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)                  # PCA দিয়ে মাত্রা হ্রাস (Dimensionality Reduction), ২টা প্রধান কম্পোনেন্ট নেওয়া
X_pca = pca.fit_transform(X_train)        # ডেটা থেকে প্রধান ফিচার বের করে নতুন ডেটাসেট তৈরি

svd = TruncatedSVD(n_components=2)        # TruncatedSVD, PCA এর মতো মাত্রা হ্রাসের আরেক পদ্ধতি, বিশেষ করে sparse ডেটার জন্য ভালো
X_svd = svd.fit_transform(X_train)        # ডেটার মাত্রা কমিয়ে নতুন ফিচার তৈরি
```

---

## ১১. Text Feature Extraction (টেক্সট থেকে ফিচার)

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = ["machine learning is fun", "sklearn is powerful"]   # টেক্সট ডেটার স্যাম্পল

cv = CountVectorizer()                                       # CountVectorizer টেক্সট থেকে শব্দের গণনা (বগ অফ ওয়ার্ডস)
X_counts = cv.fit_transform(texts)                           # টেক্সটকে সংখ্যার ম্যাট্রিক্সে রূপান্তর করা

tfidf = TfidfVectorizer()                                    # TfidfVectorizer শব্দের গুরুত্ব (TF-IDF) হিসাব করে
X_tfidf = tfidf.fit_transform(texts)                         # টেক্সটকে TF-IDF ভেক্টরে রূপান্তর করা

```

---

## ১২. Model Saving and Loading (মডেল সংরক্ষণ ও পুনরায় ব্যবহার)

```python
import joblib

joblib.dump(model, 'model.pkl')      # মডেলকে 'model.pkl' ফাইলে সংরক্ষণ করা (save)
model = joblib.load('model.pkl')     # সংরক্ষিত মডেলকে ফাইল থেকে লোড করা (load) ব্যবহার করার জন্য

```

---

## ১৩. Utility Functions (সহজ কাজের ফাংশন)

```python
model.get_params()                   # মডেলের সব বর্তমান প্যারামিটার ও সেটিংস দেখার জন্য
model.set_params(**new_params)      # নতুন প্যারামিটার দিয়ে মডেল আপডেট করার জন্য, যেমন set_params(C=1.0)
model.score(X_test, y_test)          # মডেলের পারফরম্যান্স স্কোর মাপার জন্য, যেমন accuracy বা R² স্কোর

# set_params()-এ তোমাকে অবশ্যই নতুন প্যারামিটার dictionary আকারে দিতে হবে, উদাহরণ:

model.set_params(C=0.5, max_iter=200)
```

---

