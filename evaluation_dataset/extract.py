import gzip
import os

def extract_tour_file(gz_filepath, output_filepath=None):
    """Extract a .opt.tour.gz file to .opt.tour"""
    if not gz_filepath.endswith('.gz'):
        raise ValueError("Input file must be a .gz file")
    if output_filepath is None:
        output_filepath = gz_filepath[:-3]  # Remove .gz extension
    with gzip.open(gz_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extracted {gz_filepath} to {output_filepath}")


if __name__ == "__main__":

    extracted_dir = "./extracted"
    os.makedirs(extracted_dir, exist_ok=True)
    compressed_dir = "./compressed"
    for fname in os.listdir(compressed_dir):
        if fname.endswith('.opt.tour.gz'):
            gz_path = os.path.join(compressed_dir, fname)
            output_path = os.path.join(extracted_dir, fname[:-3])
            extract_tour_file(gz_path, output_path)