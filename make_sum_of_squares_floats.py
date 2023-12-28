import csv
import random

# Define the data and the file name
# data = ["x", "y", "sum_of_square"]
file_name = "sum_of_squares_floats.csv"

# Write to the CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    samples = 0
    while samples < 10000:
        # square to random numbers between 0 and 1 and add them
        i = random.random()
        j = random.random()
        writer.writerow([i, j, (i*i)+(j*j)])
        samples += 1

print(f"The training data has been written to {file_name}")