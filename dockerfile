FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Deteksi JAVA_HOME otomatis agar tidak hardcode
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN if [ ! -d "$JAVA_HOME" ]; then \
    JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))); \
    fi

ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
