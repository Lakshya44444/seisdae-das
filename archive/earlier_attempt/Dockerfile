FROM python:3.12-slim
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends make libreoffice-writer libreoffice-impress pandoc fonts-liberation poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /study
COPY requirements*.txt constraints-linux-py312.txt ./
RUN pip install --no-cache-dir -r requirements-v2.txt -c constraints-linux-py312.txt
COPY . .
CMD ["make","reproduce"]
