pipeline {
    agent any

    environment {
        // TODO: Change these to your specific values
        STUDENT_NAME = 'aziz'  // Change this to your name (no spaces)
        STUDENT_PORT = '8110'       // Change this to your assigned port
        IMAGE_NAME = "gpu-service-${STUDENT_NAME}"
        CONTAINER_NAME = "gpu-container-${STUDENT_NAME}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📥 Checking out code from repository...'
                checkout scm
            }
        }

        stage('GPU Sanity Test') {
            steps {
                echo '🔧 Installing required dependencies for cuda_test'
                sh '''
                    # Install Python dependencies for testing
                    pip3 install --user numpy numba || true
                '''

                echo '🧪 Running CUDA sanity check...'
                sh '''
                    # Run the CUDA sanity test
                    python3 cuda_test.py

                    # Check exit code
                    if [ $? -ne 0 ]; then
                        echo "❌ CUDA sanity test failed!"
                        exit 1
                    fi

                    echo "✅ CUDA sanity test passed!"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh """
                    # Build the Docker image
                    docker build -t ${IMAGE_NAME}:latest .

                    # Tag with build number for versioning
                    docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:build-${BUILD_NUMBER}

                    echo "✅ Docker image built successfully: ${IMAGE_NAME}:latest"
                """
            }
        }

        stage('Stop Old Container') {
            steps {
                echo "🛑 Stopping and removing old container if exists..."
                sh """
                    # Stop and remove existing container (ignore errors if not exists)
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true

                    echo "✅ Old container cleaned up"
                """
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container with GPU support..."
                sh """
                    # Deploy the container with GPU access
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        --gpus all \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        -p 8000:8000 \
                        --restart unless-stopped \
                        ${IMAGE_NAME}:latest

                    # Wait for container to be healthy
                    echo "⏳ Waiting for container to be ready..."
                    sleep 5

                    # Check if container is running
                    if [ \$(docker ps -q -f name=${CONTAINER_NAME}) ]; then
                        echo "✅ Container ${CONTAINER_NAME} is running"
                        docker ps -f name=${CONTAINER_NAME}
                    else
                        echo "❌ Container failed to start!"
                        docker logs ${CONTAINER_NAME}
                        exit 1
                    fi
                """
            }
        }

        stage('Health Check') {
            steps {
                echo "🏥 Running health checks..."
                sh """
                    # Wait a bit more for service to fully start
                    sleep 10

                    # Test health endpoint
                    echo "Testing /health endpoint..."
                    curl -f http://localhost:${STUDENT_PORT}/health || exit 1

                    # Test GPU info endpoint
                    echo "Testing /gpu-info endpoint..."
                    curl -f http://localhost:${STUDENT_PORT}/gpu-info || exit 1

                    echo "✅ All health checks passed!"
                """
            }
        }

        stage('Cleanup Old Images') {
            steps {
                echo "🧹 Cleaning up old Docker images..."
                sh """
                    # Remove dangling images
                    docker image prune -f

                    # Keep only last 3 builds
                    docker images ${IMAGE_NAME} --format "{{.Tag}}" | \
                        grep "build-" | \
                        sort -t '-' -k2 -n -r | \
                        tail -n +4 | \
                        xargs -I {} docker rmi ${IMAGE_NAME}:{} || true

                    echo "✅ Cleanup completed"
                """
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            echo "🌐 Service is available at: http://10.90.90.100:${STUDENT_PORT}"
            echo "📊 Metrics available at: http://10.90.90.100:${STUDENT_PORT}/metrics"
            echo "🏥 Health check: http://10.90.90.100:${STUDENT_PORT}/health"
            echo "💻 GPU info: http://10.90.90.100:${STUDENT_PORT}/gpu-info"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
            sh """
                echo "Container logs:"
                docker logs ${CONTAINER_NAME} || true

                echo "Container status:"
                docker ps -a -f name=${CONTAINER_NAME} || true
            """
        }
        always {
            echo "🧾 Pipeline finished."
            echo "Build #${BUILD_NUMBER} completed at \$(date)"
        }
    }
}