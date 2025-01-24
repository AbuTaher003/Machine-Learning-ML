# Batch Machine Learning

Batch Machine Learning এমন একটি পদ্ধতি যেখানে মডেল ট্রেনিং এবং প্রেডিকশন প্রক্রিয়া ব্যাচ বা বড় ডাটাসেটের উপর ভিত্তি করে করা হয়। এই পদ্ধতিটি সাধারণত স্ট্যাটিক ডাটা এবং নির্দিষ্ট সময়ের মধ্যে আপডেট করা প্রয়োজন এমন ডাটাসেটের জন্য ব্যবহৃত হয়।

---

## কীভাবে কাজ করে:

1. **Static Data:** মডেল ট্রেনিংয়ের সময় একটি নির্দিষ্ট ডাটাসেট ব্যবহার করা হয় যা একবারে প্রসেস করা হয়।
2. **Model Training:** পুরো ডাটাসেট থেকে মডেল শেখে এবং প্যাটার্ন বের করে।
3. **Deployment:** মডেলটি ট্রেনিং শেষে ডেপ্লয় করা হয় এবং নতুন ডাটার জন্য পূর্বানুমান করতে ব্যবহৃত হয়।
4. **Periodic Updates:** ডাটাবেস পরিবর্তিত হলে মডেল পুনরায় ট্রেনিংয়ের মাধ্যমে আপডেট করা হয়।

---

## বৈশিষ্ট্য:

- **Static Nature:** ডাটাসেট স্ট্যাটিক হওয়ায় এটি real-time ডাটার জন্য ব্যবহার করা যায় না।
- **Periodic Training:** নতুন ডাটা এলে মডেল আবার পুরো ডাটাসেটের উপর ট্রেনিং করা হয়।
- **High Computational Cost:** বড় ডাটাসেটের জন্য বেশি প্রসেসিং পাওয়ার প্রয়োজন হয়।

---

## সুবিধা:

1. **Simple to Implement:** মডেল একবার ট্রেনিং করার পর ব্যবহার করা সহজ।
2. **Effective for Large Datasets:** বড় ডাটাসেটের জন্য কার্যকর।
3. **Suitable for Historical Data:** পূর্বের ডাটার উপর ভিত্তি করে ভালো কাজ করে।

---

## অসুবিধা:

1. **Not Suitable for Real-Time Data:** লাইভ বা স্ট্রিমিং ডাটার জন্য উপযুক্ত নয়।
2. **Time-Consuming:** প্রতিবার মডেল আপডেট করতে সময় লাগে।
3. **Overfitting Risk:** স্ট্যাটিক ডাটার উপর মডেল overfit করতে পারে।

---

## বাস্তব জীবনের উদাহরণ:

- **Fraud Detection:** ব্যাঙ্ক ট্রানজেকশনের ডাটার উপর ভিত্তি করে জালিয়াতি সনাক্ত করা।
- **Customer Churn Prediction:** গ্রাহক ডাটার উপর ভিত্তি করে ভবিষ্যতে কে সেবাটি বন্ধ করবে তা পূর্বানুমান।
- **Weather Prediction:** ঐতিহাসিক আবহাওয়ার ডাটা বিশ্লেষণ।

---

## Code Example (Python):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Features
y = [0, 0, 1, 1]  # Labels

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Code Explanation:

1. **Data Preparation:** ডাটা প্রিপেয়ার করে Train এবং Test সেটে ভাগ করা হয়েছে।
2. **Model Training:** Logistic Regression ব্যবহার করে মডেল ট্রেনিং করা হয়েছে।
3. **Prediction:** মডেল টেস্ট ডাটার উপর পূর্বানুমান করেছে।
4. **Evaluation:** Accuracy মেট্রিক দিয়ে মডেলের পারফরম্যান্স পরিমাপ করা হয়েছে।

---



