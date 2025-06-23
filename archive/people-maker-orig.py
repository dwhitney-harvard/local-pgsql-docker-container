import csv
import random
from faker import Faker
from datetime import datetime

fake = Faker()
output_file = "people.csv"
num_rows = 100_000

with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header (excluding 'id' since it's auto-generated)
    writer.writerow(['person_id', 'first_nm', 'last_nm', 'birth_dt', 'mdm_person_id'])

    for _ in range(num_rows):
        person_id = f"PID{random.randint(1000000, 9999999)}"
        mdm_person_id = f"MDM{random.randint(1000000, 9999999)}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
        writer.writerow([person_id, first_name, last_name, birth_date, mdm_person_id])
