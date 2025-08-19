FROM apache/airflow:2.7.3

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
# Install compatible versions
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.6 \
    sqlalchemy==1.4.46 \
    polars \
    scikit-learn==1.2.2 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    joblib==1.3.1