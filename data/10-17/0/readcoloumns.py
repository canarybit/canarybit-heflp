import csv

# Replace 'your_file.csv' with the path to your CSV file
file_path = '/Users/ahdesperado/thysiphus/canarybit-heflp/data/10-17/0/X_train.csv'

with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    # Read the first row (header or first data row)
    first_row = next(reader, None)
    if first_row:
        print(f"Number of columns: {len(first_row)}")
    else:
        print("The file is empty.")