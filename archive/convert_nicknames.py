import csv

input_file = "names.csv"
output_file = "converted_names.csv"

with open(input_file, newline='') as infile, open(output_file, "w", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    # Read and write the header
    header = next(reader)
    writer.writerow(header)

    # Process remaining rows
    for row in reader:
        if not row:
            continue  # skip empty lines
        canonical = row[0]
        nicknames = row[1:]
        joined_nicknames = ",".join(nicknames)
        writer.writerow([canonical, joined_nicknames])
