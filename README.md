

# Face-Embedding Search in PostgreSQL Docker Container

This repo shows how to:
1. Spin up a local PostgreSQL + pgvector Docker container.  
2. Ingest random “people” with Faker-generated metadata and CLIP image embeddings.  
3. Run a Streamlit app that does vector search (and hybrid text+image search) for face matches.  

—  
## Table of Contents  
- [Architecture](#architecture)  
- [Prerequisites](#prerequisites)  
- [Getting Started](#getting-started)  
- [Project Structure](#project-structure)  
- [Debugging & Logging](#debugging--logging)  
- [Long-Term Maintenance](#long-term-maintenance)  

## Architecture  
1. **PostgreSQL + pgvector**  
   • A Docker container hosts a `people_with_faces` table with a `VECTOR(512)` column.  
2. **Ingestion Script** (`people_ingest_embed.py`)  
   • Uses Faker for fake names (multi-locale) and PIL + CLIP to compute face embeddings.  
3. **Vector Search** (`vector_search.py`)  
   • Wraps a `<->` distance query in pgvector, returning the top‐K nearest faces.  
4. **Hybrid Re-ranking** (`hybrid_search.py` + `app-hybrid-search.py`)  
   • Combines CLIP image similarity with simple text/ID/email matching and a small scoring model.  
5. **Streamlit Front-End**  
   • Upload an image, run image→vector, query Postgres, re-rank, and show the top match.  

## Prerequisites  
- Docker & Docker Compose (or plain Docker)  
- Python 3.12  
- `git clone` this repo & `cd` into it  
- Optional GPU + CUDA for faster CLIP inference  

## Getting Started  

### Launch Postgres + pgvector

```bash
   docker-compose up -d
```

### Create & activate virtualenv

```bash
python -m venv venv
# . venv/Scripts/activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Ingest sample data

```bash
python people_ingest_embed.py
```

This will:

- Create the people_with_faces table if needed
- Generate 1,000 fake people with images, embeddings and metadata
- Insert them into Postgres

### Run the Streamlit app

```bash
streamlit run app-hybrid-search.py
```

Open your browser at http://localhost:8501. Upload a headshot and see your top face match.

## Project Structure

- **embedding_cache.py** : Caches CLIP image→vector calls in-memory
- **vector_search.py** : Runs the pgvector <-> nearest‐neighbor SQL query
- **extract_features.py** : (For batch ML) builds text/image features for training a scoring model
- **generate_training_pairs.py** : Synthesizes positive/negative duplicate pairs for model training
- **hybrid_search.py** : Simple re-ranking combining text and image similarity
- **app-hybrid-search.py** : Streamlit app that ties it all together
- **people_ingest_embed.py** : Faker-based ingestion into Postgres + pgvector

## Debugging & Logging

### Enable Python Logging

Add at top of each script:

```bash
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)
```

Replace print(...) calls with logger.info(...), logger.error(...) etc.

### Inspect Database Logs

```bash
docker logs pgvector-demo
```

### Streamlit Debug

Enable Watcher logs in ~/.streamlit/config.toml:

```bash
[logger]
level = "debug"
```

Or run with STREAMLIT_LOG_LEVEL=debug streamlit run …

### Step-through in VS Code

- Set breakpoints in vector_search.py or app-hybrid-search.py.
- Use the built-in debugger to inspect vec_literal, SQL params, DataFrame contents.

## Long-Term Maintenance

### Schema Migrations

Adopt a tool like Alembic to version your table definitions.

### CI/CD & Tests

- Write unit tests for vector_search.find_similar_faces, mocking psycopg2
- Add integration tests using a Docker-in-Docker PostgreSQL

### Logging Rotation

Switch to logging.handlers.RotatingFileHandler or a centralized solution (e.g. ELK).

### Performance Monitoring

- Track query latency via Postgres pg_stat_statements
- Benchmark CLIP latency & batch sizes

### Security & Credentials

- Move DB creds into environment variables or Azure Key Vault
- Restrict network access

With these building blocks, you can prototype and productionize a face‐matching service that combines image embeddings, fast vector search, and simple text logic—all backed by a local Docker container.

## APPENDIX Local PGSQL Docker Container

Reference: [https://harvardmed.atlassian.net/browse/SOLAR-51](https://harvardmed.atlassian.net/browse/SOLAR-51)

## How to use this

```bash
docker-compose up -d
```

### Enable Extensions in SQL

Once the container is running:

```bash
docker exec -it phonetic_pg psql -U postgres -d mydb
```

In the PostgreSQL shell:

```sql
-- For phonetic functions
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- For similarity matching (optional but useful)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Phonetic Search Usage

```sql
-- Soundex
SELECT soundex('Robert'), soundex('Rupert'); -- should be similar

-- Metaphone (example)
SELECT metaphone('Smith', 4), metaphone('Smythe', 4);

-- Trigram similarity
SELECT 'bob' % 'robert';  -- true if similar
```

## Load Into PostgreSQL

```bash
# Step 1: Download the CSV
wget https://raw.githubusercontent.com/carltonnorthern/nickname-and-diminutive-names-lookup/master/names.csv

# Step 2: format the data correctly
convert_nicknames.py

# Step 3: Copy it into your container
docker cp converted_names.csv phonetic_pg:/tmp/names.csv

# Step 4: Load it in SQL
docker exec -it phonetic_pg psql -U postgres -d mydb

# Inside the psql prompt:
DROP TABLE IF EXISTS nicknames;

CREATE TABLE nicknames (
  nickname TEXT,
  canonical TEXT
);

COPY nicknames(nickname, canonical)
FROM '/tmp/names.csv'
DELIMITER ','
CSV HEADER
QUOTE '"';
```

## Make the people table

people_with_faces_to_pg.py

```sql
DROP TABLE IF EXISTS people;

CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    person_id INT,
    first_nm TEXT,
    last_nm TEXT,
    birth_dt DATE,
    mdm_person_id BIGINT,
    email_address VARCHAR(320)
);
```

## DEV ONLY: create fake people file

This needs to be updated to use only the db

### Setup

```bash
python3 -m venv venv
pip install faker
python3 people_with_faces_to_pg.py
docker cp people_with_faces.csv phonetic_pg:/tmp/people.csv
docker exec -it phonetic_pg psql -U postgres -d mydb
```

### Finish

Load people with faces data into the people table

```bash
COPY people(person_id, first_nm, last_nm, birth_dt, mdm_person_id, email_address)
FROM '/tmp/people.csv'
DELIMITER ','
CSV HEADER;
```

Test

```sql
SELECT id,
       person_id,
       first_nm,
       last_nm,
       birth_dt,
       mdm_person_id,
       email_address
FROM public.people
LIMIT 25;
```

## Create name enrichment view

```sql
CREATE OR REPLACE VIEW people_enriched AS
SELECT
    p.*,

    -- Fuzzy search: Soundex and Metaphone
    soundex(p.first_nm) AS first_nm_soundex,
    soundex(p.last_nm) AS last_nm_soundex,

    metaphone(p.first_nm, 4) AS first_nm_metaphone,
    metaphone(p.last_nm, 4) AS last_nm_metaphone,

    -- -- Trigram similarity to a target string
    -- similarity(LOWER(p.first_nm), 'robert') AS first_nm_similarity_to_robert,

    -- Nickname mapping
    n.canonical AS normalized_first_name

FROM people p
LEFT JOIN nicknames n ON LOWER(p.first_nm) = LOWER(n.nickname);
```

```sql
SELECT id,
       person_id,
       first_nm,
       last_nm,
       birth_dt,
       mdm_person_id,
       email_address,
       first_nm_soundex,
       last_nm_soundex,
       first_nm_metaphone,
       last_nm_metaphone,
       normalized_first_name
FROM public.people_enriched
LIMIT 10;
```

## Installation of tds_fdw

```bash
sudo apt update
sudo apt install postgresql-server-dev-all freetds-dev git make gcc

git clone https://github.com/tds-fdw/tds_fdw.git
cd tds_fdw

make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
sudo make install PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
```

### Enable it in psql

```sql
CREATE EXTENSION tds_fdw;
```

### Create a connection

```sql
CREATE SERVER mssql_svr
  FOREIGN DATA WRAPPER tds_fdw
  OPTIONS (
    servername 'your_sql_server_hostname',
    port '1433',
    database 'your_database_name'
  );

CREATE USER MAPPING FOR postgres
  SERVER mssql_svr
  OPTIONS (username 'sqluser', password 'yourpassword');
```
