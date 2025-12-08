FROM python:3.13

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -U \
    flask \
    google-genai \
    sqlalchemy \
    psycopg2-binary

EXPOSE 8000

CMD ["python", "main.py"]
