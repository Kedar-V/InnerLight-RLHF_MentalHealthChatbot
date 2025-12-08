FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir flask google-genai  sqlalchemy psycopg2-binary
EXPOSE 8000
# Run the Flask development server. For production, consider replacing this with
# a proper WSGI server (gunicorn/uWSGI) and a process manager.
CMD ["python", "main.py", "serve"]
