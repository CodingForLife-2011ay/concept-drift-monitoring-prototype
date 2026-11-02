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
# Create the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")
# New incoming batches

X_batch1 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch1 = [0, 0, 1]


X_batch2 = [
    [6, 95],
    [1, 50],
    [4, 85]
]

y_batch2 = [1, 0, 1]


X_batch3 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch3 = [1, 1, 0]


X_batch4 = [
    [2, 60],
    [3, 70],
    [5, 90]
]

y_batch4 = [1, 1, 0]
# Generate predictions

predictions_batch1 = model.predict(X_batch1)
predictions_batch2 = model.predict(X_batch2)
predictions_batch3 = model.predict(X_batch3)
predictions_batch4 = model.predict(X_batch4)
# Calculate accuracy for each batch

accuracy_batch1 = sum(predictions_batch1 == y_batch1) / len(y_batch1)
accuracy_batch2 = sum(predictions_batch2 == y_batch2) / len(y_batch2)
accuracy_batch3 = sum(predictions_batch3 == y_batch3) / len(y_batch3)
accuracy_batch4 = sum(predictions_batch4 == y_batch4) / len(y_batch4)
# Display batch results

print("========== BATCH RESULTS ==========")

print("Batch 1 accuracy:", accuracy_batch1 * 100, "%")
print("Batch 2 accuracy:", accuracy_batch2 * 100, "%")
print("Batch 3 accuracy:", accuracy_batch3 * 100, "%")
print("Batch 4 accuracy:", accuracy_batch4 * 100, "%")

# Monitoring system

threshold = 0.60

batch_accuracies = [
    accuracy_batch1,
    accuracy_batch2,
    accuracy_batch3,
    accuracy_batch4
]

print("\n========== MONITORING ALERTS ==========")

for i, accuracy in enumerate(batch_accuracies):

    if accuracy < threshold:
        print("Batch", i + 1, "- WARNING: Low accuracy!")
    else:
        print("Batch", i + 1, "- Performance acceptable.")

# Detect repeated low accuracy

low_accuracy_count = 0

print("\n========== REPEATED LOW ACCURACY ==========")

for accuracy in batch_accuracies:

    if accuracy < threshold:
        low_accuracy_count += 1
    else:
        low_accuracy_count = 0

    if low_accuracy_count >= 2:
        print("ALERT: Repeated low accuracy detected!")
# Decide whether an investigation is needed

investigation_needed = low_accuracy_count >= 2

if investigation_needed:
    print("\nInvestigation required!")
    print("Checking the incoming data for possible changes...")
else:
    print("\nNo investigation required.")
# Investigation dataset

import pandas as pd

new_data = {
    "Study Hours": [2, 3, 5, 6, 1, 4, 3],
    "Attendance": [60, 70, 90, 95, 50, 85, 83],
    "Passed": [1, 1, 0, 0, 1, 1, 0]
}

new_df = pd.DataFrame(new_data)

print("\n========== INVESTIGATION DATA ==========")
print(new_df)
# Compare original and new input data

original_data = {
    "Study Hours": [2, 3, 5, 6, 1, 4, 3],
    "Attendance": [60, 70, 90, 95, 50, 85, 83],
    "Passed": [1, 0, 1, 1, 0, 1, 1]
}

original_df = pd.DataFrame(original_data)

print("\n========== INPUT DATA COMPARISON ==========")

print("Study hours unchanged:",
      original_df["Study Hours"].equals(new_df["Study Hours"]))

print("Attendance unchanged:",
      original_df["Attendance"].equals(new_df["Attendance"]))
# Compare original and new pass rates

original_pass_rate = original_df["Passed"].mean()
new_pass_rate = new_df["Passed"].mean()

print("\n========== PASS RATE COMPARISON ==========")

print("Original pass rate:",
      round(original_pass_rate * 100, 2), "%")

print("New pass rate:",
      round(new_pass_rate * 100, 2), "%")

print("Difference:",
      round((original_pass_rate - new_pass_rate) * 100, 2),
      "percentage points")
# Connect monitoring to investigation

if investigation_needed:

    print("\n========== INVESTIGATION FINDINGS ==========")

    print("Repeated low accuracy was detected.")

    print("Study hours unchanged:",
          original_df["Study Hours"].equals(new_df["Study Hours"]))

    print("Attendance unchanged:",
          original_df["Attendance"].equals(new_df["Attendance"]))

    print("Original pass rate:",
          round(original_pass_rate * 100, 2), "%")

    print("New pass rate:",
          round(new_pass_rate * 100, 2), "%")

    print("Possible change in the relationship between inputs and outcomes.")

else:
    print("\nInvestigation was not triggered.")
# Noise investigation

noisy_df = original_df.copy()

# Simulate one unusual outcome
noisy_df.loc[0, "Passed"] = 0

noisy_pass_rate = noisy_df["Passed"].mean()

print("\n========== NOISE INVESTIGATION ==========")

print("Original pass rate:",
      round(original_pass_rate * 100, 2), "%")

print("Noisy pass rate:",
      round(noisy_pass_rate * 100, 2), "%")

print("Noise pass rate difference:",
      round((original_pass_rate - noisy_pass_rate) * 100, 2),
      "percentage points")
# Count changed outcomes

concept_drift_changes = (
    original_df["Passed"] != new_df["Passed"]
).sum()

noise_changes = (
    original_df["Passed"] != noisy_df["Passed"]
).sum()

print("\n========== CHANGE COMPARISON ==========")

print("Changed outcomes in concept drift:",
      concept_drift_changes)

print("Changed outcomes in noise:",
      noise_changes)
# Final system conclusion

print("\n========== FINAL SYSTEM CONCLUSION ==========")

if investigation_needed:
    print("Repeated low model accuracy was detected.")

    if concept_drift_changes > noise_changes:
        print("The investigation found more changed outcomes")
        print("in the concept drift simulation than in the noise simulation.")

    print("Possible concept drift should be investigated further.")

else:
    print("No repeated low accuracy was detected.")
    print("No investigation was triggered.")