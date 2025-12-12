FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY distributed_ml_system ./distributed_ml_system

CMD ["python", "-m", "distributed_ml_system.app.scripts.run_distributed_train"]
