from sklearn.tree import DecisionTreeClassifier

# Training examples: [hours studied, attendance]
X_train = [
    [2, 60],
    [3, 70],
    [5, 90],
    [6, 95],
    [1, 50],
    [4, 85],
    [3, 83]
]

# Correct labels: 0 = did not pass, 1 = passed
y_train = [1, 0, 1, 1, 0, 1,1]

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Teach the model using the examples
model.fit(X_train, y_train)

# Ask the model to predict for a new example
prediction = model.predict([[1, 50]])
print("Prediction:", prediction[0])



# Test the model on the original pattern
X_before = [
    [2, 60],
    [3, 70],
    [5, 90],
    [6, 95],
    [1, 50],
    [4, 85]
]

y_before = [0, 0, 1, 1, 0, 1]

# Test the model after the concept changes
X_after = [
    [2, 60],
    [3, 70],
    [5, 90],
    [6, 95],
    [1, 50],
    [4, 85]
]

y_after = [0, 0, 0, 1, 0, 0]

# Make predictions WITHOUT retraining the model
predictions_before = model.predict(X_before)
predictions_after = model.predict(X_after)

# Calculate accuracy
accuracy_before = sum(
    predictions_before == y_before
) / len(y_before)

accuracy_after = sum(
    predictions_after == y_after
) / len(y_after)

print("Accuracy before drift:", accuracy_before)
print("Accuracy after drift:", accuracy_after)


import matplotlib.pyplot as plt

# Store the accuracy results
labels = ["Before drift", "After drift"]
accuracies = [accuracy_before, accuracy_after]

# Create a bar chart
plt.bar(labels, accuracies)

# Add a title and labels
plt.title("Model Accuracy Before and After Concept Drift")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Display the chart
plt.show()



# The model's predictions on the original inputs
predictions = model.predict(X_before)

# Case 1: Original correct answers
y_normal = y_before

# Case 2: Simulate one unexpected result (noise)
y_noise = [0, 0, 1, 1, 0, 0]

# Case 3: Simulate a changed concept
y_drift = y_after

# Calculate accuracy in each situation
accuracy_normal = sum(
    predictions == y_normal
) / len(y_normal)

accuracy_noise = sum(
    predictions == y_noise
) / len(y_noise)

accuracy_drift = sum(
    predictions == y_drift
) / len(y_drift)

import matplotlib.pyplot as plt

situations = ["Normal", "Noise", "Concept Drift"]

accuracies = [
    accuracy_normal * 100,
    accuracy_noise * 100,
    accuracy_drift * 100
]

plt.bar(situations, accuracies)

plt.title("Comparing Model Accuracy")
plt.xlabel("Situation")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)
plt.show()

# 4. NEW: Add monitoring code below
drop = (accuracy_before - accuracy_after) * 100
threshold = 10

print("Accuracy before:", accuracy_before * 100, "%")
print("Accuracy after:", accuracy_after * 100, "%")
print("Accuracy drop:", drop, "percentage points")

if drop >= threshold:
    print("Warning: Investigate possible drift")
else:
    print("No warning from this threshold")

# Display the results
print("\n--- Phase 6 Results ---")
print("Normal accuracy:", accuracy_normal)
print("Accuracy with noise:", accuracy_noise)
print("Accuracy after concept drift:", accuracy_drift)

