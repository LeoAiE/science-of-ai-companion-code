# A minimal Docker image to serve the FastAPI model from 28.4.
# Build:  docker build -t bc-service .
# Run:    docker run -p 8000:8000 bc-service

# 1. Start from a slim, pinned base image: a known Python on a small Linux.
FROM python:3.13.18-slim

# 2. Set the working directory inside the image.
WORKDIR /app

# 3. Copy ONLY requirements first, then install. This layer is cached and reused
#    on later builds unless requirements change, so rebuilds are fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Now copy the application code (changes often, so it comes after the deps layer).
COPY . .

# 5. The build gate of Section 28.6: load 04_serving.py as the server will and check its
#    label for one known held-out tumor; an import error or a wrong label fails the build.
RUN python 06_containerization.py

# 6. Copy the numbered teaching script to a name an ordinary import statement can use
#    (import 04_serving is a syntax error, because the name starts with a digit) and
#    start the server on that name. Importing it is cheap: the script keeps its demo,
#    benchmark and test client under `if __name__ == "__main__":`, so the import only
#    trains the model and defines the app.
RUN cp 04_serving.py serving.py

# 7. Document the port and define the start command for the container.
EXPOSE 8000
CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]
