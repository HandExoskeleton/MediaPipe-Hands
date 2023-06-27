import csv
import numpy as np


# Create a 1D matrix (NumPy array) with 21 columns
matrix = np.zeros(21)

# Prompt user to enter file
csv_file = input("Enter file name: ")

# Open the CSV file for reading
file = open(csv_file, 'r')

# Create a CSV reader object
reader = csv.reader(file)

# Skip the header row
next(reader)

# Initialize previous row
previous_row = None

# Read and process each row
for row in reader:
    # Check if previous row exists
    if previous_row is not None:
        # Subtract respective values from each cell tuple in the current row
        result = []
        for curr_cell, prev_cell in zip(row, previous_row):
            curr_tuple = tuple(map(int, curr_cell.strip('()').split(',')))
            prev_tuple = tuple(map(int, prev_cell.strip('()').split(',')))

            # Perform subtraction based on tuple length
            if len(curr_tuple) == len(prev_tuple):
                diff = tuple(curr_val - prev_val for curr_val, prev_val in zip(curr_tuple, prev_tuple))
            else:
                diff = None  # Handle the case when tuple lengths are different

            result.append(diff)

        # Print the result
        print(result)

    # Assign the current row as the previous row for the next iteration
    previous_row = row

# Close the file
file.close()
