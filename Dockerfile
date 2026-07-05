FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./main.py
COPY configs/configuration_docker_example.json ./configs/configuration_docker_example.json

RUN pip install --no-cache-dir .

CMD ["python", "main.py", "--help"]
