# Use this script to extract all `.gz` files from the `tsp` folder into the `tsp_dataset_extracted` folder
# Also creates the output folder if it does not exist

import gzip
import os

# Extract a single .gz file
def extract_gz_file(gz_filepath, output_filepath):
    with gzip.open(gz_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extracted {gz_filepath} to {output_filepath}")


def extract_all_gz_files(directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            gz_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename[:-3])  # Remove .gz extension
            extract_gz_file(gz_path, output_path)

# Usage
extract_all_gz_files("./compressed", "./extracted")