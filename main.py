import os
import csv


def create_annotations(directory_path, gcs_bucket_path, output_csv_path):
    """
    Generates a CSV annotation file from a directory of images.

    The script iterates over files in the specified directory, extracts labels
    from the filenames, and creates a CSV file mapping the GCS path of each
    image to its labels.

    Args:
        directory_path (str): The local path to the directory containing the images.
        gcs_bucket_path (str): The base GCS path (e.g., 'gs://your-bucket/images').
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    # Define the header for the CSV file
    header = ['gcs_path', 'labels']
    rows = []

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Iterate over each file in the specified directory
    for filename in os.listdir(directory_path):
        # Process only files with supported image extensions
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            try:
                # --- Label Extraction Logic ---
                # Example filename: 111__A_5_50.jpeg or 88__AD_5_100.jpg

                # Get the part after the double underscore
                label_part = filename.split('__')[1]

                # Get the part before the next underscore, which contains the labels
                label_string = label_part.split('_')[0]

                # The labels are the individual characters in the label_string
                labels = list(label_string)

                # --- Path and Row Creation ---

                # Get just the directory name for the GCS path, making it more robust
                directory_name = os.path.basename(directory_path)

                # Construct the full GCS path for the file
                gcs_file_path = f"{gcs_bucket_path}/{directory_name}/{filename}"

                # Create the row, starting with the GCS path
                row = [gcs_file_path]
                # Add the individual labels as separate elements to the row
                row.extend(labels)

                # Add the constructed row to our list of all rows
                rows.append(row)

            except IndexError:
                print(f"Warning: Skipping file '{filename}' due to unexpected format.")

    # Write the collected data to the CSV file
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the header
            # csv_writer.writerow(header) # You can uncomment this if you want a header row

            # Write the data rows
            csv_writer.writerows(rows)
        print(f"Successfully created annotation file at '{output_csv_path}' with {len(rows)} entries.")
    except IOError:
        print(f"Error: Could not write to file '{output_csv_path}'. Check permissions.")


if __name__ == '__main__':
    # --- Configuration ---
    # The local directory where your image files are stored.
    IMAGE_DIRECTORY = "/Users/valeria/Documents/magnus/original_imgs"

    # The base path for your Google Cloud Storage bucket.
    GCS_BUCKET = "gs://magnus-images"

    # The name of the output CSV file.
    OUTPUT_CSV = "annotations_orig.csv"

    # --- Execution ---
    # Check if the directory exists before running
    if not os.path.exists(IMAGE_DIRECTORY):
        print(f"Directory '{IMAGE_DIRECTORY}' not found.")
        print("Please create it or run the 'setup_test_environment.py' script first.")
    else:
        create_annotations(IMAGE_DIRECTORY, GCS_BUCKET, OUTPUT_CSV)

