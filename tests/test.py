import numpy as np

# Parameters
prediction = 1        # The model's prediction
target = 0            # The actual target value
old_w = 1             # Initial weight
learning_rate = 0.1   # Learning rate for weight update
x = 1                 # Input feature
num_iterations = 20   # Number of iterations for weight updates

for i in range(num_iterations):
    # Calculate the error (loss)
    loss = 0.5 * (prediction - target) ** 2

    # Calculate the gradient
    gradient_d = prediction - target  # Gradient with respect to the prediction
    gradient = gradient_d * x  # Gradient with respect to the weight

    # Update the weight
    new_w = old_w - learning_rate * gradient

    # Assign new_w to old_w
    old_w = new_w

    # Print the results for each iteration
    print(f"Iteration {i+1}:")
    print(f"  Loss: {loss}")
    print(f"  Gradient: {gradient}")
    print(f"  Updated weight: {old_w}")

    # Update the prediction based on the new weight, if applicable
    # Assuming prediction is influenced by the weight (e.g., prediction = old_w * x)
    prediction = old_w * x
