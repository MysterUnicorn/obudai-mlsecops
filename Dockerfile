FROM python:3.12.3-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


WORKDIR /app
COPY src/* /app
RUN useradd -u 1001 unpriviledged_user && \
    chown unpriviledged_user:unpriviledged_user /app


USER unpriviledged_user
ENV PYTHONPATH=/app
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
