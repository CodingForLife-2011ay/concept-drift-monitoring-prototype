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