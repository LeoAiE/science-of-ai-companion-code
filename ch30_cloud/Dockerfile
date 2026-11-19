# The shared serving image for the Chapter 30 lab (see 30.10.2 and Chapter 28
# for what each instruction does). Build with model.joblib and app.py beside it:
#   docker build -t breast-cancer-api:1.0 .
FROM python:3.13.18-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.joblib app.py ./
# Cloud Run injects PORT (8080 unless you set another port). Container Apps and
# ECS Express Mode send traffic to the port you declare instead, so their deploy
# commands declare 8080, the default set here (30.10.2).
ENV PORT=8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
