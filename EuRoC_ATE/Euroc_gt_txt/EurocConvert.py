import csv

# Code used for converting the ground truth CSV files into the 7 column text format used in the estimation files.

targetFile = "V_203"
# Specify the input CSV file and output TXT file
csv_file_path = f"535project\\535project\\Euroc_gt\\{targetFile}.csv"
txt_file_path = f"535project\\535project\\Euroc_gt_txt\\{targetFile}.txt"

# Specify the columns you want to keep (assuming 0-based indexing)
columns_to_keep = [1, 2, 3, 4, 5, 6, 7]

# Read CSV and write selected columns to TXT
with open(csv_file_path, 'r') as csv_file, open(txt_file_path, 'w') as txt_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        selected_columns = [row[i] for i in columns_to_keep]
        txt_file.write(' '.join(selected_columns) + '\n')
