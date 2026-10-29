# Financial Fraud Detection Platform

An end-to-end machine learning platform for detecting potentially fraudulent financial transactions using a trained ML model, FastAPI, real-time event processing, and automated testing.

## 🚀 Live Application

* **Frontend:** https://financial-fraud-detection-nu.vercel.app
* **Backend API:** https://financial-fraud-detection-tf41.onrender.com
* **API Documentation:** https://financial-fraud-detection-tf41.onrender.com/docs

## 📌 Project Overview

The Financial Fraud Detection Platform analyzes transaction features and predicts whether a transaction may be fraudulent. It combines machine learning with a REST API and a web interface to make fraud prediction accessible through an interactive application.

The project also includes Kafka-based transaction event processing, feature-store integration, model evaluation, and automated tests.

## ✨ Key Features

* Machine learning-based fraud prediction
* Interactive web interface for submitting transaction data
* FastAPI backend with REST endpoints
* Automatic API documentation using Swagger UI
* Kafka transaction producer and consumer components
* Feature-store integration using Feast
* Model evaluation and saved model artifacts
* Automated testing with pytest
* GitHub Actions continuous integration
* Frontend deployment through Vercel
* Backend deployment through Render

## 🛠️ Technology Stack

**Programming:** Python, JavaScript

**Machine Learning:** Scikit-learn, PyTorch, XGBoost

**Backend:** FastAPI, Uvicorn

**Frontend:** React, Vite

**Streaming:** Apache Kafka, kafka-python

**Feature Store:** Feast

**Data Processing:** Pandas, NumPy, PyArrow

**Experiment Tracking:** MLflow

**Testing:** Pytest, HTTPX

**Deployment and DevOps:** GitHub Actions, Docker, Vercel, Render

## 🏗️ Project Structure

```text
financial-fraud-detection/
├── .github/
│   └── workflows/
├── artifacts/
├── backend/
├── feature_store/
├── frontend/
├── k8s/
├── ml/
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── feature_store.yaml
├── requirements.txt
├── runtime.txt
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mymudh/financial-fraud-detection.git
cd financial-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the backend

Check the backend entry point and required environment variables in the repository, then start the FastAPI application using the appropriate module.

For example, if `backend/main.py` exposes the FastAPI instance as `app`:

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Configure the frontend's `VITE_API_URL` environment variable to point to your backend URL.

## 🔄 Transaction Event Processing

The project includes Kafka producer and consumer components for transaction-event processing.

Publish a transaction using the producer's supported command-line options:

```bash
python -m backend.kafka_producer --help
```

Run the consumer using the documented configuration and an available Kafka broker:

```bash
python -m backend.kafka_consumer
```

Kafka must be configured and running before testing event processing.

## 🧪 Automated Testing

Run the test suite from the project root:

```bash
python -m pytest
```

To run the Kafka producer tests:

```bash
python -m pytest backend/test_kafka_prediction.py -v
```

Use the test file names present in the repository if they differ.

The project includes automated checks for API behavior, prediction functionality, transaction processing, and feature-store functionality.

## 📊 Model Evaluation

The machine learning pipeline includes model training and evaluation components. Evaluation metrics and trained model artifacts are stored in the repository's `artifacts/` directory.

Model performance depends on the dataset, feature definitions, preprocessing, and evaluation methodology. Predictions should be validated against representative transaction data before practical use.

## 🔐 Security and Limitations

* Do not commit API keys, passwords, or private environment variables.
* Validate transaction inputs before prediction.
* Use appropriate authentication and access controls before production use.
* Treat model predictions as decision-support signals rather than definitive proof of fraud.
* Test the system with representative data and monitor false positives and false negatives.

## 🔮 Future Improvements

* Real-time fraud monitoring dashboard
* Explainable AI for transaction-level predictions
* Model drift detection and retraining
* Improved transaction investigation workflows
* Performance monitoring and production observability

## 👨‍💻 Author

**Shaik Maimood**

GitHub: https://github.com/Mymudh

## 📄 License

Choose and add an appropriate open-source license before distributing the project for reuse.
