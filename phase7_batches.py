from sklearn.tree import DecisionTreeClassifier

# Training data
X_train = [
    [2, 60],
    [3, 70],
    [5, 90],
    [6, 95],
    [1, 50],
    [4, 85],
    [3, 83]
]

y_train = [1, 0, 1, 1, 0, 1, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Batch 1: Original student records

X_batch1 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch1 = [0, 0, 1]

# Make predictions for Batch 1
predictions_batch1 = model.predict(X_batch1)

# Calculate accuracy
accuracy_batch1 = sum(predictions_batch1 == y_batch1) / len(y_batch1)

# Display results
print("Batch 1 predictions:", predictions_batch1)
print("Batch 1 actual results:", y_batch1)
print("Batch 1 accuracy:", accuracy_batch1 * 100, "%")

# Batch 2: New student records

X_batch2 = [
    [6, 95],
    [1, 50],
    [4, 85]
]

y_batch2 = [1, 0, 1]
# Make predictions for Batch 2
predictions_batch2 = model.predict(X_batch2)

# Calculate accuracy
accuracy_batch2 = sum(predictions_batch2 == y_batch2) / len(y_batch2)

# Display results
print("Batch 2 predictions:", predictions_batch2)
print("Batch 2 actual results:", y_batch2)
print("Batch 2 accuracy:", accuracy_batch2 * 100, "%")

# Batch 3: Simulated concept drift

X_batch3 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch3 = [1, 1, 0]
# Make predictions for Batch 3
predictions_batch3 = model.predict(X_batch3)

# Calculate accuracy
accuracy_batch3 = sum(predictions_batch3 == y_batch3) / len(y_batch3)

# Display results
print("Batch 3 predictions:", predictions_batch3)
print("Batch 3 actual results:", y_batch3)
print("Batch 3 accuracy:", accuracy_batch3 * 100, "%")

# Batch 4: More data after the simulated change
X_batch4 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch4 = [1, 1, 0]

# Make predictions
predictions_batch4 = model.predict(X_batch4)

# Calculate accuracy
accuracy_batch4 = sum(predictions_batch4 == y_batch4) / len(y_batch4)

# Display results
print("Batch 4 predictions:", predictions_batch4)
print("Batch 4 actual results:", y_batch4)
print("Batch 4 accuracy:", accuracy_batch4 * 100, "%")

# Store all batch accuracies
batch_accuracies = [
    accuracy_batch1,
    accuracy_batch2,
    accuracy_batch3,
    accuracy_batch4
]

# Set an example accuracy threshold
threshold = 0.60

# Check each batch
for i, accuracy in enumerate(batch_accuracies):
    print("Batch", i + 1, "accuracy:", accuracy * 100, "%")

    if accuracy < threshold:
        print("Warning: Low accuracy detected!")
    else:
        print("Performance is acceptable.")

# Count consecutive low-accuracy batches
low_accuracy_count = 0

for accuracy in batch_accuracies:

    if accuracy < threshold:
        low_accuracy_count += 1
    else:
        low_accuracy_count = 0

    if low_accuracy_count >= 2:
        print("Alert: Repeated low accuracy detected!")