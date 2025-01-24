# Types of Machine Learning

Machine Learning (ML) তিনটি প্রধান ভাগে বিভক্ত, এবং প্রতিটি ভাগের নিজস্ব বৈশিষ্ট্য এবং প্রক্রিয়া রয়েছে। এগুলো হলো:

## 1. Supervised Learning
Supervised Learning-এ ডাটার সাথে "লেবেল" দেয়া থাকে। এর মানে হলো, ডাটার প্রতিটি ইনপুটের জন্য সঠিক আউটপুটও জানানো হয়। মডেলটি শেখে কীভাবে ইনপুট থেকে আউটপুট বের করতে হয়।

### কীভাবে কাজ করে:
1. **Training Data:** লেবেল করা ডাটা (যেমন, "এটি বিড়াল," "এটি কুকুর") মডেলে দেয়া হয়।
2. **Model Training:** মডেল ডাটার সাথে আউটপুটের সম্পর্ক শিখে।
3. **Prediction:** নতুন ডাটা এলে মডেল পূর্বে শেখা জ্ঞান ব্যবহার করে আউটপুট ভবিষ্যদ্বাণী করে।

### উদাহরণ:
- **Email Spam Detection:** মডেল শিখে নেয় কোন ইমেইল স্প্যাম এবং কোনটি নয়।
- **Image Classification:** মডেলকে বিভিন্ন ছবি দেখিয়ে শেখানো হয় কোনটি বিড়াল আর কোনটি কুকুর।
- **Medical Diagnosis:** রোগ নির্ণয়ের জন্য রোগীর ডাটা ব্যবহার করে রোগ শনাক্ত করা।

### Algorithm উদাহরণ:
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVMs)

---

## 2. Unsupervised Learning
Unsupervised Learning-এ ডাটার কোন লেবেল থাকে না। মডেল শুধু ডাটার মধ্যে লুকানো pattern এবং সম্পর্ক খুঁজে বের করে। এটি exploratory analysis-এ ব্যবহৃত হয়।

### কীভাবে কাজ করে:
1. **Input Data:** মডেলকে শুধু ইনপুট ডাটা দেয়া হয়, কোন লেবেল ছাড়াই।
2. **Pattern Discovery:** মডেল ডাটার মধ্যে ক্লাস্টার বা গ্রুপ খুঁজে বের করে।
3. **Insights:** গ্রুপ বা প্যাটার্ন থেকে গুরুত্বপূর্ণ তথ্য বের করা হয়।

### উদাহরণ:
- **Customer Segmentation:** মডেল শিখে কোন কাস্টমারদের অভ্যাস একরকম।
- **Anomaly Detection:** ডাটার মধ্যে অস্বাভাবিক কিছু শনাক্ত করা।
- **Market Basket Analysis:** কোন প্রোডাক্ট একসাথে কেনা হয় তা বিশ্লেষণ করা।

### Algorithm উদাহরণ:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Autoencoders

---

## 3. Reinforcement Learning
Reinforcement Learning-এ মডেল একটি environment-এ কাজ করে এবং প্রতিটি action-এর জন্য reward বা penalty পায়। মডেল শেখে কীভাবে সবচেয়ে বেশি reward পেতে হয়।

### কীভাবে কাজ করে:
1. **Agent এবং Environment:** মডেল (agent) একটি environment-এ কাজ করে।
2. **Action:** Agent environment-এর উপর ভিত্তি করে action নেয়।
3. **Reward:** Environment থেকে reward বা penalty পাওয়া যায়।
4. **Learning:** Reward-এর উপর ভিত্তি করে agent ভবিষ্যতে ভালো action নিতে শেখে।

### উদাহরণ:
- **Game Playing:** AI মডেল শেখে কীভাবে একটি গেমে জিততে হয়।
- **Robotics:** একটি রোবট শেখে কীভাবে একটি নির্দিষ্ট কাজ সঠিকভাবে করতে হয়।
- **Self-driving Cars:** গাড়ি শেখে কীভাবে সঠিক রাস্তা অনুসরণ করতে হয়।

### Algorithm উদাহরণ:
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- SARSA

---

## Comparison Table:
| Feature              | Supervised Learning      | Unsupervised Learning  | Reinforcement Learning |
|----------------------|--------------------------|-------------------------|-------------------------|
| **Labelled Data**    | Yes                      | No                      | No                      |
| **Output Known?**    | Yes                      | No                      | No                      |
| **Goal**             | Predict output           | Find patterns           | Maximize rewards        |
| **Example**          | Spam detection           | Customer segmentation   | Game AI                 |

---

### বাস্তব জীবনের উদাহরণ:

- **Supervised Learning:**
  - মোবাইল ফোনে ছবি তোলার সময় AI ফিচার ছবির সাবজেক্ট (মানুষ, প্রাণী) সনাক্ত করে।

- **Unsupervised Learning:**
  - ই-কমার্স সাইটে পণ্য সাজানোর জন্য কাস্টমারের কেনার অভ্যাস বিশ্লেষণ।

- **Reinforcement Learning:**
  - AlphaGo মডেল গেম খেলে এবং নিজে নিজে শেখে কিভাবে জিততে হয়।

---

