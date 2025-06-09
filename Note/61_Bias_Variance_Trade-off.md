For Hand notes you can check here 👉🏻 [click](https://drive.google.com/file/d/11vYMt1PuERAo5S28Pqdh-JFvYw6eqHhN/view?usp=drive_link)

# Bias-Variance Tradeoff

The **Bias-Variance Tradeoff** is a fundamental concept in supervised machine learning that explains the tradeoff between a model’s ability to generalize to unseen data versus its performance on the training data.

---

##  What is Bias?

- Bias is the **error due to overly simplistic assumptions** in the learning algorithm.
- High bias models:
  - Underfit the data
  - Miss relevant relations between input and output
  - Have poor performance on both training and test data

**Example:** A linear model trying to fit a complex non-linear dataset.

---

##  What is Variance?

- Variance is the **error due to high sensitivity to small fluctuations** in the training set.
- High variance models:
  - Overfit the training data
  - Learn noise as if it were a signal
  - Perform well on training data but poorly on test data

**Example:** A very deep decision tree trained on a small dataset.

---

##  The Tradeoff

| Bias      | Variance  | Model Behavior     |
|-----------|-----------|--------------------|
| High      | Low       | Underfitting        |
| Low       | High      | Overfitting         |
| Balanced  | Balanced  | Good Generalization |

A good model should **balance bias and variance** to minimize total prediction error.

---

##  Total Error Formula

Total Error = Bias² + Variance + Irreducible Error


- **Bias²**: Error from erroneous assumptions in the model.
- **Variance**: Error from sensitivity to training data fluctuations.
- **Irreducible Error**: Error due to noise in the data.

---

##  Visualization

Underfitting <--- Balanced ---> Overfitting
High Bias High Variance
| |
Low Training & Test Acc High Train, Low Test Acc

---

##  How to Control Bias and Variance?

| Action                             | Effect                      |
|------------------------------------|-----------------------------|
| Increase model complexity          | ↓ Bias, ↑ Variance          |
| Decrease model complexity          | ↑ Bias, ↓ Variance          |
| Use more training data             | ↔ Bias, ↓ Variance          |
| Regularization (L1, L2)            | ↔ Bias, ↓ Variance          |
| Feature selection / simplification | ↑ Bias, ↓ Variance          |
| Ensemble methods (e.g., Bagging)   | ↔ Bias, ↓ Variance          |

---

##  Goal of a Good Model

- Not too simple (to avoid high bias)
- Not too complex (to avoid high variance)
- Should generalize well on unseen data

---

## 📚 References

- [Deep Learning Book - Chapter 5](https://www.deeplearningbook.org/)
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
- [CS229 Lecture Notes - Stanford](https://cs229.stanford.edu/)
