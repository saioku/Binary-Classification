import pandas as pd

# Load the submission.csv file
input_file = "submission1.csv"
output_file = "submission.csv"

# Read the CSV file
data = pd.read_csv(input_file)

# Ensure the 'id' column is treated as a numeric column for sorting
data['id'] = pd.to_numeric(data['id'], errors='coerce')

# Sort the data by 'id' in ascending order
sorted_data = data.sort_values(by='id')

# Save the sorted data to a new CSV file
sorted_data.to_csv(output_file, index=False)

print(f"Sorted file saved as {output_file}.")
