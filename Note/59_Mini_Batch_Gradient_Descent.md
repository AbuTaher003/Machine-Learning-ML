Go to the code, where you will find the respective code for this topic. Explanation is provided through comments within the code to ensure clarity and understanding.

You can access the code using the following link:
[View Code Here]()

---






# Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a popular optimization algorithm that blends the advantages of both **Batch Gradient Descent** and **Stochastic Gradient Descent (SGD)**. It helps achieve faster convergence while maintaining stability and scalability.

---

##  What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize the cost (loss) function in machine learning by updating parameters (like weights in neural networks) in the opposite direction of the gradient.

---

##  Types of Gradient Descent

| Type                  | Description |
|-----------------------|-------------|
| **Batch GD**          | Uses the entire training set for each update. |
| **Stochastic GD**     | Uses one random data point for each update. |
| **Mini-Batch GD**     | Uses a small subset (mini-batch) of data points for each update. |

---

##  Why Mini-Batch Gradient Descent?

- Faster convergence than batch gradient descent
- Less noise than stochastic gradient descent
- Efficient use of hardware (especially GPUs)
- Enables vectorized operations for speed

---

## ⚙️ How Mini-Batch GD Works

1. **Shuffle** the training data.
2. **Divide** the data into small batches of fixed size (e.g., 32, 64, 128).
3. For each epoch:
   - For each mini-batch:
     - Compute the gradient of the loss function using that mini-batch
     - Update the model parameters using the gradient

---

##  Algorithm Pseudocode

Initialize parameters θ randomly
Repeat until convergence:
Shuffle training data
Divide data into mini-batches of size m
For each mini-batch:
Compute gradient ∇J(θ; mini_batch)
θ = θ - α * ∇J(θ; mini_batch)


Where:
- `θ` = parameters
- `α` = learning rate
- `∇J(θ)` = gradient of the cost function

---

##  Choosing Mini-Batch Size

- Common batch sizes: **32**, **64**, **128**
- Small batch size = more updates per epoch (faster learning, more noise)
- Large batch size = fewer updates per epoch (slower learning, more stable)

> Tip: Start with 32 or 64 and tune as needed based on model and dataset.

---

##  Advantages

- Balances stability and speed
- Can leverage hardware parallelism
- Less noisy than SGD
- More scalable than batch GD

---

##  Disadvantages

- Still introduces some noise (but less than SGD)
- Requires tuning of mini-batch size
- Not always as stable as full batch GD

---

## 📚 References

- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [CS231n - Stanford University](https://cs231n.github.io/optimization-1/)
