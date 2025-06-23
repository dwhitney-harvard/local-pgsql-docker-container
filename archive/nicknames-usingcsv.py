# File based script to load a CSV file containing nickname mappings and normalize names
# using the mappings. The CSV file should have two columns: 'canonical' and 'nickname'.
# The script reads the CSV file, creates a dictionary for nickname mappings, and provides
# a function to normalize names based on the mappings.
import csv

def load_nickname_map(csv_path="converted_names.csv"):
    nickname_dict = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nickname = row['nickname'].strip().lower()
            canonical = row['canonical'].strip().lower()
            nickname_dict[nickname] = canonical
    return nickname_dict

def normalize_name(name, nickname_map):
    return nickname_map.get(name.lower(), name.lower())
