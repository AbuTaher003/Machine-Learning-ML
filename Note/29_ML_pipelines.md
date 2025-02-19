Go to the code, where you will find the respective code for this topic. Explanation is provided through comments within the code to ensure clarity and understanding.

You can access the code using the following link:
[View Code Here]()

---
---



# Scikit-Learn পাইপলাইন: সহজভাবে বোঝা

Scikit-Learn-এর **Pipeline** হল একটা দারুণ টুল, যেটা **মেশিন লার্নিং-এর কাজকে সহজ করে**। এটা ডাটা প্রসেসিং আর মডেলিং-এর ধাপগুলো **একসাথে এনে** কোডকে **পরিষ্কার আর ইউজফুল** বানায়।

## 📌 পাইপলাইন কেন দরকার?
- **প্রিপ্রসেসিং আর মডেলিং** একসাথে করা যায়।
- **ডাটা লিকেজ এড়ানো যায়**, কারণ ট্রেনিং ডাটার উপরেই ট্রান্সফরমেশন হয়।
- **কোড ছোট, সহজ আর পুনরায় ব্যবহারযোগ্য হয়।**
- **GridSearchCV দিয়ে সহজে টিউনিং করা যায়।**

---

## 🛠️ Scikit-Learn ইনস্টল করো
```bash
pip install scikit-learn
```

---

## 🚀 একটা সাধারণ পাইপলাইন কেমন হয়?
একটা পাইপলাইনে **ডাটা প্রসেসিং** আর **মডেল** থাকে। নিচে `StandardScaler` আর `LogisticRegression` দিয়ে একটা পাইপলাইন বানানো হলো।

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

এখানে প্রথমে **ডাটা স্কেল হবে**, তারপর **Logistic Regression মডেল ট্রেন হবে**।

---

## ⚙️ পাইপলাইন দিয়ে পুরো ওয়ার্কফ্লো
চল, **ডাটা লোড, প্রসেসিং, আর মডেল ট্রেনিং** সহ একটা কমপ্লিট পাইপলাইন করি।

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f'মডেলের অ্যাকুরেসি: {accuracy:.2f}')
```

---

## 🔄 `GridSearchCV` দিয়ে পাইপলাইন টিউন করা
যদি **হাইপারপ্যারামিটার টিউন করতে চাও**, তাহলে `GridSearchCV` ব্যবহার করতে পারো।

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'সেরা প্যারামিটার: {grid_search.best_params_}')
print(f'সেরা স্কোর: {grid_search.best_score_:.2f}')
```

---

## 🔄 কাস্টম ট্রান্সফরমার দিয়ে পাইপলাইন
নিজের মতো **কাস্টম প্রসেসিং স্টেপ** বানিয়ে পাইপলাইনে যোগ করা যায়।

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

pipeline = Pipeline([
    ('custom_transform', CustomTransformer()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
```

---

## 📝 পাইপলাইনের সুবিধা
| ফিচার | সুবিধা |
|---------|---------|
| **ওয়ার্কফ্লো অটোমেটেড হয়** | ম্যানুয়ালি ধাপ কমাতে হয় |
| **ডাটা লিকেজ হয় না** | ট্রেন-টেস্ট স্প্লিট ঠিক থাকে |
| **GridSearchCV টিউনিং সহজ হয়** | পারফরম্যান্স অপটিমাইজ করা যায় |
| **কোড পরিষ্কার আর ছোট হয়** | একই পাইপলাইন আবার ইউজ করা যায় |

---

## 🎯 শেষ কথা
- **পাইপলাইন ইউজ করলে** কোড পরিষ্কার হয় আর সহজে মডেল ডেপলয় করা যায়।
- **GridSearchCV দিয়ে হাইপারপ্যারামিটার টিউন করা সহজ হয়।**
- **কাস্টম ট্রান্সফরমার বানিয়ে নিজের মতো প্রসেসিং করা যায়।**


