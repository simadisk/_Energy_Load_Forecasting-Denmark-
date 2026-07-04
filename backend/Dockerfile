# 1. Παίρνουμε μια καθαρή, ελαφριά έκδοση της Python
FROM python:3.10-slim

# 2. Φτιάχνουμε έναν φάκελο /app μέσα στον εικονικό server
WORKDIR /app

# 3. Αντιγράφουμε τη λίστα με τις βιβλιοθήκες και τις κάνουμε install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Αντιγράφουμε τον κώδικά μας (το main.py)
COPY . .

# 5. Ανοίγουμε την πόρτα 8000 για να μπαίνει ο κόσμος
EXPOSE 8000

# 6. Η εντολή που θα τρέχει το API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]