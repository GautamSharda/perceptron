import csv

# Define the data and the file name
# data = ["x", "y", "sum"]
file_name = "addition.csv"

# Write to the CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    for i in range(100):
        for j in range(100):
            writer.writerow([i, j, i+j])

print(f"The data has been written to {file_name}")