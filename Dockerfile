FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/tmp/huggingface PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py passages.json ./
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
