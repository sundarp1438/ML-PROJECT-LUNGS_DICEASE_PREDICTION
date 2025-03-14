pipeline {
    agent any

    environment {
        PYTHON_VERSION = "3.12"
        VENV_DIR = "venv"
        MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
        DOCKER_REGISTRY = "sundarp1985"
        DOCKER_IMAGE = "lung-disease-api"
        K8S_NAMESPACE = "mlops"
    }

    stages {
        stage('Checkout Repository') {
            steps {
                script {
                    echo "📥 Pulling latest code from GitHub..."
                    checkout scm
                }
            }
        }

        stage('Set up Python Environment') {
            steps {
                script {
                    echo "🐍 Setting up virtual environment..."
                    sh "python${PYTHON_VERSION} -m venv ${VENV_DIR}"
                    sh ". ${VENV_DIR}/bin/activate && pip install --upgrade pip"
                    sh ". ${VENV_DIR}/bin/activate && pip install -r requirements.txt"
                }
            }
        }

        stage('Load and Preprocess Data') {
            steps {
                script {
                    echo "📊 Running data loading and preprocessing..."
                    sh ". ${VENV_DIR}/bin/activate && python src/stages/load_data.py --config=params.yaml"
                    sh ". ${VENV_DIR}/bin/activate && python src/stages/featurize_data.py --config=params.yaml"
                    sh ". ${VENV_DIR}/bin/activate && python src/stages/data_split.py --config=params.yaml"
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    echo "🎯 Training ML model..."
                    sh ". ${VENV_DIR}/bin/activate && python mlflow_pipeline.py --config=params.yaml"
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    echo "📈 Evaluating ML model..."
                    sh ". ${VENV_DIR}/bin/activate && python evaluate_model.py --config=params.yaml"
                }
            }
        }

        stage('Run Data Drift Monitoring') {
            steps {
                script {
                    echo "📉 Running data drift monitoring..."
                    sh ". ${VENV_DIR}/bin/activate && python lung_disease_drift.py"
                }
            }
        }

        stage('Track and Push to DVC') {
            steps {
                script {
                    echo "📦 Tracking artifacts in DVC..."
                    sh ". ${VENV_DIR}/bin/activate && dvc add data/processed"
                    sh ". ${VENV_DIR}/bin/activate && dvc add models"
                    sh ". ${VENV_DIR}/bin/activate && dvc push"
                }
            }
        }

        stage('Commit and Push Changes') {
            steps {
                script {
                    echo "🚀 Committing and pushing changes to GitHub..."
                    sh """
                    git config --global user.email "sundarp1@yahoo.com"
                    git config --global user.name "sundarp1438"
                    git add .
                    git commit -m "Automated update: Processed data, models, and reports"
                    git push origin main
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "🐳 Building Docker image..."
                    sh "docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest -f Dockerfile ."
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    echo "📤 Pushing Docker image to Docker Hub..."
                    sh "echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin"
                    sh "docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    echo "🚢 Deploying to Kubernetes..."
                    
                    sh "kubectl apply -f k8s/deployment.yaml -n ${K8S_NAMESPACE}"
                    sh "kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}"
                    
                    echo "✅ Waiting for deployment to be ready..."
                    sh "kubectl rollout status deployment/lung-disease-api -n ${K8S_NAMESPACE}"
                }
            }
        }
    }

    post {
        success {
            script {
                echo "✅ Pipeline completed successfully!"
                // Slack or email notification can be added here
            }
        }
        failure {
            script {
                echo "❌ Pipeline failed!"
                // Notify the team (e.g., Slack, email)
            }
        }
    }
}
