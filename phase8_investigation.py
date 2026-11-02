# Import pandas
import pandas as pd

# Original student data
original_data = {
    "Study Hours": [2, 3, 5, 6, 1, 4, 3],
    "Attendance": [60, 70, 90, 95, 50, 85, 83],
    "Passed": [1, 0, 1, 1, 0, 1, 1]
}

# Convert the data into a DataFrame
original_df = pd.DataFrame(original_data)

# Display the original data
print("--- Original Student Data ---")
print(original_df)

# New student data
new_data = {
    "Study Hours": [2, 3, 5, 6, 1, 4, 3],
    "Attendance": [60, 70, 90, 95, 50, 85, 83],
    "Passed": [1, 1, 0, 0, 1, 1, 0]
}

# Convert into a DataFrame
new_df = pd.DataFrame(new_data)

# Display the new data
print("\n--- New Student Data ---")
print(new_df)

# Compare original and new input data
print("\n--- Comparing Input Data ---")

print("Original average study hours:",
      original_df["Study Hours"].mean())

print("New average study hours:",
      new_df["Study Hours"].mean())

print("Original average attendance:",
      original_df["Attendance"].mean())

print("New average attendance:",
      new_df["Attendance"].mean())

# Compare passing rates

original_pass_rate = original_df["Passed"].mean()
new_pass_rate = new_df["Passed"].mean()

print("\n--- Comparing Pass Rates ---")

print("Original pass rate:", original_pass_rate * 100, "%")
print("New pass rate:", new_pass_rate * 100, "%")

# Calculate the change in pass rate

pass_rate_difference = original_pass_rate - new_pass_rate

print("\n--- Pass Rate Difference ---")
print("Decrease in pass rate:",
      pass_rate_difference * 100, "percentage points")

# Investigate possible concept drift

if pass_rate_difference > 0:
    print("\nPossible change in the relationship between inputs and outcomes.")

if original_pass_rate != new_pass_rate:
    print("The pass rates are different.")

if (original_df["Study Hours"].equals(new_df["Study Hours"])
        and original_df["Attendance"].equals(new_df["Attendance"])):
    print("The input data stayed exactly the same.")

# Simulate noise: change one student's outcome

noisy_data = original_df.copy()

# Change the outcome of one student
noisy_data.loc[0, "Passed"] = 0

# Display the noisy data
print("\n--- Noisy Student Data ---")
print(noisy_data)

# Calculate noisy pass rate
noisy_pass_rate = noisy_data["Passed"].mean()

print("\n--- Noise Comparison ---")
print("Original pass rate:", original_pass_rate * 100, "%")
print("Noisy pass rate:", noisy_pass_rate * 100, "%")

print("Difference:",
      (original_pass_rate - noisy_pass_rate) * 100,
      "percentage points")

print("\n--- Final Investigation Summary ---")

print("Original pass rate:", original_pass_rate * 100, "%")
print("Concept drift pass rate:", new_pass_rate * 100, "%")
print("Noise pass rate:", noisy_pass_rate * 100, "%")

# Final Investigation Report

print("\n========== INVESTIGATION REPORT ==========")

print("\n1. INPUT DATA")
if (original_df["Study Hours"].equals(new_df["Study Hours"])
        and original_df["Attendance"].equals(new_df["Attendance"])):
    print("Input data remained unchanged.")

print("\n2. CONCEPT DRIFT SIMULATION")
print("Original pass rate:", round(original_pass_rate * 100, 2), "%")
print("New pass rate:", round(new_pass_rate * 100, 2), "%")
print("Pass rate difference:",
      round((original_pass_rate - new_pass_rate) * 100, 2),
      "percentage points")

print("\n3. NOISE SIMULATION")
print("Original pass rate:", round(original_pass_rate * 100, 2), "%")
print("Noisy pass rate:", round(noisy_pass_rate * 100, 2), "%")

print("\n4. CONCLUSION")
print("The experiments demonstrate different types of changes.")
print("Further investigation is needed to identify real-world drift.")