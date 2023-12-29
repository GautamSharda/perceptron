import csv
import random
import os

# Path to directory that stores training data
DATA_DIRECTORY = 'data'

def make_addition_floats():
    # Define the data and the file name
    # data = ["x", "y", "sum_of_square"]
    file_path = os.path.join(DATA_DIRECTORY, 'addition_floats.csv')

    # Write to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        samples = 0
        while samples < 10000:
            # add 2 random numbers between 0 and 1
            i = random.random()
            j = random.random()
            writer.writerow([i, j, i+j])
            samples += 1

    print(f"The training data has been written to {file_path}")

def make_addition():
    # Define the data and the file name
    # data = ["x", "y", "sum"]
    file_path = os.path.join(DATA_DIRECTORY, 'addition.csv')

    # Write to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(100):
            for j in range(100):
                writer.writerow([i, j, i+j])

    print(f"The training data has been written to {file_path}")


def make_sum_of_squares_floats():
    # Define the data and the file name
    # data = ["x", "y", "sum_of_square"]
    file_path = os.path.join(DATA_DIRECTORY, 'sum_of_squares_floats.csv')

    # Write to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        samples = 0
        while samples < 10000:
            # square to random numbers between 0 and 1 and add them
            i = random.random()
            j = random.random()
            writer.writerow([i, j, (i*i)+(j*j)])
            samples += 1

    print(f"The training data has been written to {file_path}")

def make_sum_of_squares():
    # Define the data and the file name
    # data = ["x", "y", "sum_of_square"]
    file_path = os.path.join(DATA_DIRECTORY, 'sum_of_squares.csv')

    # Write to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(100):
            for j in range(100):
                writer.writerow([i, j, (i*i)+(j*j)])

    print(f"The training data has been written to {file_path}")

make_sum_of_squares()