# Build the Docker image
build:
    docker build -t sentiment-analyzer .

# Run the Docker container
run:
    docker run -p 8501:8501 sentiment-analyzer

# Stop and remove the container
clean:
    docker stop $(docker ps -q --filter ancestor=sentiment-analyzer) || true
    docker rm $(docker ps -a -q --filter ancestor=sentiment-analyzer) || true