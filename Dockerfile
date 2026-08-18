FROM ghcr.io/oracle/oraclelinux8-instantclient:21

RUN dnf install -y \
    python39 \
    python39-pip \
    python39-devel \
    gcc \
    gcc-c++ \
    make

WORKDIR /app

RUN dnf install -y glibc-langpack-ko glibc-langpack-en

ENV PYTHON_ORACLEDB_DRIVER_TYPE=thin

RUN pip3 install --upgrade pip

COPY requirements.txt .
# Install faiss-cpu separately (works with pip, no conda needed)
RUN pip3 install faiss-cpu
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]