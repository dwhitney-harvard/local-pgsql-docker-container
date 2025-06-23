import csv
import random
import base64
from faker import Faker
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

fake = Faker()
output_file = "people_with_headshots.csv"
num_rows = 100_000

def generate_headshot_base64(name):
    """Generate a base64-encoded dummy headshot with initials."""
    img = Image.new('RGB', (128, 128), color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
    draw = ImageDraw.Draw(img)

    initials = ''.join([n[0] for n in name.split()[:2]]).upper()
    font = ImageFont.load_default()
    text_w, text_h = draw.textsize(initials, font=font)
    draw.text(((128 - text_w) / 2, (128 - text_h) / 2), initials, fill=(0, 0, 0), font=font)

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['person_id', 'first_nm', 'last_nm', 'birth_dt', 'mdm_person_id', 'headshot_b64'])

    for _ in range(num_rows):
        person_id = f"PID{random.randint(1000000, 9999999)}"
        mdm_person_id = f"MDM{random.randint(1000000, 9999999)}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        full_name = f"{first_name} {last_name}"
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
        headshot = generate_headshot_base64(full_name)
        writer.writerow([person_id, first_name, last_name, birth_date, mdm_person_id, headshot])

print(f"✅ Done! Generated {num_rows} rows with headshots in {output_file}")
