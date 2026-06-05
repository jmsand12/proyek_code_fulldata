FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    procps \
    wget \
    && rm -rf /var/lib/lib/lists/*

# Set JAVA_HOME secara eksplisit + fallback
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment && \
    echo "export JAVA_HOME=$JAVA_HOME" >> /etc/profile && \
    echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verifikasi Java tersedia saat build
RUN java -version && echo "JAVA OK"
RUN python3 -c "import pyspark; print('PySpark OK')"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
