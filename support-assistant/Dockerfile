FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache them between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the assistant code and committed policy corpus.
COPY . .

ENV MOCK_LLM=1
EXPOSE 8000

# Serve the FastAPI app locally on the required port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
