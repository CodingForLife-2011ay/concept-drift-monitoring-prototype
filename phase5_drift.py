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
#print("Prediction:", prediction[0])


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

