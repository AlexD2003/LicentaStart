import csv

# Input and output file paths
input_file = "model.txt"  # Change this to the actual file name
output_file = "labels.csv"

# Define column headers
headers = ["image_id", "tissue_type", "abnormality", "severity", "x", "y", "radius"]

# Process the data
structured_data = []

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()

        # Ensure the line isn't empty
        if not parts:
            continue

        # Extract mandatory fields
        image_id = parts[0]
        tissue_type = parts[1]
        abnormality = parts[2] if len(parts) > 2 else "NORM"
        severity = parts[3] if len(parts) > 3 and abnormality != "NORM" else ""
        x = parts[4] if len(parts) > 4 else ""
        y = parts[5] if len(parts) > 5 else ""
        radius = parts[6] if len(parts) > 6 else ""

        # Append to structured data
        structured_data.append([image_id, tissue_type, abnormality, severity, x, y, radius])

# Write the CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)  # Write headers
    writer.writerows(structured_data)  # Write data rows

print(f"CSV file '{output_file}' successfully created!")
