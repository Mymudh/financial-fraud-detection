# Financial Fraud Detection Platform

### An End-to-End Machine Learning System for Financial Fraud Detection

An intelligent fraud detection platform designed to identify potentially fraudulent financial transactions using machine learning, real-time data streaming, feature engineering, and model monitoring. The project combines a FastAPI backend, React dashboard, Kafka streaming, Feast feature store, and MLflow experiment tracking in an integrated architecture.

**Developer:** Shaik Maimood
**Repository:** [financial-fraud-detection](https://github.com/Mymudh/financial-fraud-detection)

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation and Setup](#installation-and-setup)
* [Running the Application](#running-the-application)
* [API Documentation](#api-documentation)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Model Evaluation](#model-evaluation)
* [Testing](#testing)
* [Docker](#docker)
* [Monitoring and Retraining](#monitoring-and-retraining)
* [Deployment](#deployment)
* [Future Improvements](#future-improvements)
* [Author](#author)

---

## Overview

Financial fraud is a major challenge for banks, payment providers, and financial institutions. Traditional transaction-processing systems can struggle to identify suspicious patterns quickly and adapt to changing fraud behaviour.

This project aims to provide a modular fraud detection platform that combines machine learning with supporting data infrastructure to analyse transaction information and generate predictions.

The platform is designed around the following components:

* **Machine learning:** Predict whether a transaction may be fraudulent.
* **REST API:** Expose prediction and health-check endpoints.
* **Interactive dashboard:** Provide a visual interface for interacting with the application.
* **Event streaming:** Support transaction-event processing with Apache Kafka.
* **Feature management:** Integrate Feast for feature retrieval.
* **Experiment tracking:** Record machine learning experiments with MLflow.
* **Model monitoring:** Analyse data drift and support a retraining workflow.
* **Containerisation:** Package the backend with Docker.
* **Deployment configuration:** Include Kubernetes deployment resources.

The repository brings these components together in a single project, with production deployment and end-to-end integration requiring separate verification.

---

## Key Features

### 1. Machine Learning-Based Fraud Detection

* Transaction classification using a trained machine learning model.
* Support for transaction feature validation.
* Fraud probability and prediction outputs.
* Evaluation using appropriate classification metrics.

### 2. FastAPI Backend

* REST API for transaction predictions.
* Health-check endpoint.
* Request validation using Pydantic.
* Interactive API documentation.
* Modular backend components.

### 3. Interactive React Dashboard

* Web-based interface for the fraud detection platform.
* Transaction monitoring interface.
* API connectivity indicators.
* Visual presentation of fraud detection information.

### 4. Real-Time Event Streaming

Apache Kafka provides the foundation for transaction-event streaming.

The intended streaming workflow is:

1. Receive a transaction.
2. Publish the transaction event to a Kafka topic.
3. Process the event through the consumer workflow.
4. Generate a fraud prediction.
5. Make the prediction available to the application.

End-to-end streaming behaviour depends on the producer, consumer, broker, and prediction service being configured and running together.

### 5. Feature Store Integration

Feast supports feature management and online feature retrieval.

The project includes engineered transaction features such as:

* Transaction amount
* Log-transformed amount
* Amount z-score
* Amount percentile rank
* Time-based sine transformation
* Time-based cosine transformation

The feature names supplied to the prediction model must match the features expected by the trained model.

### 6. Experiment Tracking with MLflow

MLflow supports machine learning experiment tracking, including recorded training metrics and run information.

This helps make model experiments easier to inspect and compare.

### 7. Data Drift Monitoring

The project includes scripts for detecting changes in transaction feature distributions.

The monitoring workflow can identify drift and initiate a retraining process when the configured conditions are met.

Automated retraining should be evaluated separately from automatic approval or deployment of a new model.

### 8. Graph Neural Network Support

The repository includes graph neural network-related training and prediction components intended to support graph-based fraud analysis.

The effectiveness of these components depends on the graph construction, training data, evaluation methodology, and integration with the application.

### 9. Containerisation and Deployment Configuration

* Dockerfile for packaging the API.
* Docker Compose configuration.
* Kubernetes deployment configuration.
* Automated test workflows.

Each deployment component must be configured for the target environment before production use.

---

## System Architecture

```text
                    ┌────────────────────────┐
                    │      React Frontend    │
                    │     Web Dashboard      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      FastAPI Backend   │
                    │  Validation & Routing  │
                    └───────────┬────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
       ┌────────────────────┐    ┌────────────────────┐
       │ Fraud Prediction   │    │ Feast Feature Store│
       │ ML Model           │    │ Feature Retrieval  │
       └────────────────────┘    └────────────────────┘

       ┌────────────────────┐    ┌────────────────────┐
       │ Apache Kafka       │    │ MLflow             │
       │ Transaction Events │    │ Experiment Tracking│
       └────────────────────┘    └────────────────────┘

       ┌────────────────────┐    ┌────────────────────┐
       │ Drift Monitoring   │    │ Docker / Kubernetes│
       │ Retraining Workflow│    │ Deployment         │
       └────────────────────┘    └────────────────────┘
```

**Architecture note:** This diagram represents the intended component relationships. It does not imply that every component is currently deployed or connected in a production environment.

---

## Technology Stack

| Category             | Technologies                                |
| -------------------- | ------------------------------------------- |
| Programming Language | Python, JavaScript                          |
| Machine Learning     | Scikit-learn, XGBoost                       |
| Deep Learning        | PyTorch and graph neural network components |
| Backend              | FastAPI, Uvicorn, Pydantic                  |
| Frontend             | React, Vite                                 |
| Event Streaming      | Apache Kafka                                |
| Feature Store        | Feast                                       |
| Experiment Tracking  | MLflow                                      |
| Drift Monitoring     | Evidently and custom monitoring scripts     |
| Data Processing      | Pandas, NumPy                               |
| Model Persistence    | Joblib                                      |
| Testing              | Pytest                                      |
| Containerisation     | Docker, Docker Compose                      |
| Orchestration        | Kubernetes                                  |
| Version Control      | Git, GitHub                                 |
| CI/CD                | GitHub Actions configuration                |

---

## Project Structure

```text
financial-fraud-detection/
│
├── .github/
│   └── workflows/
│
├── artifacts/
│   └── fraud_model.joblib
│
├── backend/
│   ├── main.py
│   ├── predictor.py
│   ├── feature_service.py
│   ├── kafka_producer.py
│   └── ...
│
├── datasets/
│
├── feature_store/
│   ├── feature_store.yaml
│   ├── feature_views.py
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── k8s/
│   └── deployment.yaml
│
├── ml/
│   ├── train_model.py
│   ├── train_model_mlflow.py
│   ├── feature_engineering.py
│   ├── drift_monitor.py
│   ├── auto_retrain.py
│   └── ...
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

*The structure above highlights the main components. Individual files may vary as development continues.*

---

## Installation and Setup

### Prerequisites

Install the following tools before starting:

* Python 3.11 or a compatible Python version
* Node.js and npm
* Git
* Docker Desktop
* A configured Kafka environment for streaming functionality

Kubernetes is optional for local development and is required only when using the Kubernetes deployment configuration.

### 1. Clone the Repository

```bash
git clone https://github.com/Mymudh/financial-fraud-detection.git
cd financial-fraud-detection
```

### 2. Create a Python Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 5. Verify Required Model Files

Confirm that the trained model exists:

```text
artifacts/fraud_model.joblib
```

If the model file is missing, follow the model training instructions in the `ml/` directory before starting prediction services.

---

## Running the Application

### Start the Backend

Run the following command from the project root:

```powershell
uvicorn backend.main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

### Open API Documentation

Visit:

```text
http://127.0.0.1:8000/docs
```

Use the interactive documentation to inspect the available endpoints and test supported requests.

### Start the Frontend

Open another terminal:

```powershell
cd frontend
npm run dev
```

Vite will display the local frontend address in the terminal.

The frontend must be configured to communicate with the correct backend URL.

---

## API Documentation

FastAPI generates interactive API documentation from the backend routes.

| Endpoint        | Purpose                                   |
| --------------- | ----------------------------------------- |
| `GET /health`   | Check API health                          |
| `POST /predict` | Submit a transaction for fraud prediction |
| `GET /docs`     | Open interactive API documentation        |

Additional endpoints may be available for streaming, graph prediction, feature retrieval, and case management, depending on the routes registered in the current application.

### Example Prediction Request

The exact request schema is defined by the backend's Pydantic model. A typical transaction request may contain fields such as:

```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "Amount": 149.62
}
```

This is an illustrative partial example, not a complete valid request for a model that requires `Time`, `V1` through `V28`, and `Amount`.

Use the schema displayed in `/docs` to provide all required fields.

---

## Machine Learning Pipeline

The machine learning workflow consists of the following stages:

1. **Data preparation:** Load and validate transaction data.
2. **Exploratory analysis:** Inspect transaction distributions and class imbalance.
3. **Feature engineering:** Prepare numerical and time-derived features.
4. **Model training:** Train the configured fraud classification model.
5. **Model evaluation:** Measure performance using classification metrics.
6. **Model persistence:** Save the trained model for prediction.
7. **API integration:** Load the model in the backend.
8. **Monitoring:** Track data drift and model experiments.
9. **Retraining:** Run the configured retraining workflow when monitoring conditions are satisfied.

Training and evaluation should use appropriate data splits to avoid information leakage.

---

## Model Evaluation

The project has reported the following approximate local training results:

| Metric       |   Reported Result |
| ------------ | ----------------: |
| ROC-AUC      |            0.9808 |
| PR-AUC       |            0.8741 |
| Fraud Recall | Approximately 87% |

These figures are previously reported training results, not independently verified benchmark results for the current GitHub version.

Fraud detection should be evaluated using precision, recall, F1-score, PR-AUC, confusion matrix, and false-positive rates. Accuracy alone is insufficient for an imbalanced fraud dataset.

---

## Testing

The project uses Pytest for automated tests.

Run the available tests from the repository root:

```powershell
python -m pytest -v
```

To generate a coverage report, install the coverage plugin:

```powershell
python -m pip install pytest-cov
python -m pytest --cov=backend --cov=ml --cov-report=term-missing
```

Review failing tests before treating the project as ready for deployment.

---

## Docker

Build the backend image from the project root:

```powershell
docker build -t fraud-detection-api:latest .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 fraud-detection-api:latest
```

Check the API at:

```text
http://127.0.0.1:8000/docs
```

The Docker image must include the model, required Python modules, and any files needed during application startup. Feast data, configuration files, Kafka connectivity, and other external dependencies must be configured appropriately for container execution.

---

## Monitoring and Retraining

The repository includes scripts for data drift analysis, model training, and retraining workflows.

Example commands:

```powershell
python -m ml.drift_monitor
```

```powershell
python -m ml.monitor_and_retrain
```

Use the exact script paths and arguments defined in the repository if they differ from these examples.

Before using automated retraining in a real environment, add safeguards for model validation, evaluation against a baseline, model versioning, and controlled promotion.

---

## Deployment

Deployment resources are provided for container-based deployment.

### Docker Deployment

Build and run the backend image after verifying that the Docker image contains all required application files.

### Kubernetes Deployment

The repository includes a Kubernetes manifest under `k8s/`.

Before applying it, verify:

* Kubernetes is enabled and a cluster is available.
* The container image is built and accessible to the cluster.
* The health-check path matches the API.
* Required feature-store files are present.
* Kafka and other dependencies are reachable from the cluster.
* Sensitive configuration is supplied securely.

The existence of deployment files does not itself mean that the application has been deployed publicly.

---

## Future Improvements

* Complete cloud deployment of the backend and frontend.
* Fully integrated Kafka producer-consumer prediction workflow.
* Improved graph-based fraud detection and evaluation.
* Model versioning and controlled model promotion.
* Production-grade logging, metrics, and alerting.
* Authentication, authorization, and rate limiting.
* Database-backed investigation and case management.
* Automated CI/CD with deployment gates.
* Load testing and transaction throughput benchmarking.
* Security testing and secrets management.

---

## Author

**Shaik Maimood**
AI/ML Engineering | Machine Learning | Backend Development | MLOps

GitHub: [Mymudh](https://github.com/Mymudh)

---

## Disclaimer

This project is intended for educational and demonstration purposes. Its predictions should not be used as the sole basis for real financial decisions without rigorous validation, security review, monitoring, and appropriate human oversight.
