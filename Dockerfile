FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG COMMIT_SHA=""
ARG BUILD_TIMESTAMP=""
ENV APP_COMMIT_SHA=${COMMIT_SHA}
ENV APP_BUILD_TIMESTAMP=${BUILD_TIMESTAMP}

CMD streamlit run app/ui/st_app.py --server.port=8080 --server.address=0.0.0.0

