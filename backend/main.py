import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.case_management import router as case_management_router
from backend.feature_service import get_transaction_features
from backend.gnn_predictor import get_gnn_status, predict_graph
from backend.kafka_producer import publish_transaction
from backend.predictor import predict_transaction
from backend.schemas import TransactionInput
from backend.transactions_db import (
    initialize_database,
    save_transaction,
    get_transaction_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_loaded = False

    try:
        initialize_database()
        logger.info("Transaction database initialized.")

        if not MODEL_PATH.exists():
            logger.error("Fraud model not found: %s", MODEL_PATH)
        else:
            joblib.load(MODEL_PATH)
            app.state.model_loaded = True
            logger.info("Fraud detection model loaded successfully.")

    except Exception:
        logger.exception("Application initialization failed.")

    yield


app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Real-time financial fraud detection using machine learning, "
        "feature retrieval, Kafka transaction streaming, "
        "GNN inference, SQLite transaction history, "
        "and analyst case management."
    ),
    version="1.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(case_management_router)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    threshold: float
    transaction_id: int | None = None


class GNNPredictionRequest(BaseModel):
    features: list[list[float]]
    edges: list[list[int]] = []
    node_index: int = 0


@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API is running",
        "version": "1.6.0",
        "docs": "/docs",
        "health": "/health",
        "transaction_history": "/transactions/history",
    }


@app.get("/health")
def health_check():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "model_loaded": False,
                "message": "Fraud model file not found.",
            },
        )

    try:
        joblib.load(MODEL_PATH)

        return {
            "status": "healthy",
            "model_loaded": True,
            "message": "Fraud detection model is available.",
        }

    except Exception as exc:
        logger.exception("Model health check failed.")

        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "model_loaded": False,
                "message": "Fraud model could not be loaded.",
            },
        ) from exc


@app.get("/features/{transaction_id}")
def get_features(transaction_id: int):
    try:
        features = get_transaction_features(transaction_id)

        return {
            "transaction_id": transaction_id,
            "features": features,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Feature retrieval failed.")

        raise HTTPException(
            status_code=500,
            detail="Feature retrieval failed. Check the server logs.",
        ) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionInput):
    try:
        transaction_data = data.model_dump()
        result = predict_transaction(transaction_data)

        response_data = {
            "prediction": int(result["prediction"]),
            "label": str(result["label"]),
            "fraud_probability": float(result["fraud_probability"]),
            "threshold": float(result["threshold"]),
        }

        transaction_id = save_transaction(
            transaction_data,
            response_data,
        )

        logger.info(
            "Transaction %s saved with prediction %s.",
            transaction_id,
            response_data["label"],
        )

        return {
            **response_data,
            "transaction_id": transaction_id,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Fraud prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc


@app.get("/transactions/history")
def transaction_history(limit: int = 100):
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 500.",
        )

    try:
        records = get_transaction_history(limit)

        return {
            "count": len(records),
            "transactions": records,
        }

    except Exception as exc:
        logger.exception("Transaction history retrieval failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve transaction history.",
        ) from exc


@app.post("/predict/{transaction_id}")
def predict_with_features(
    transaction_id: int,
    data: TransactionInput,
):
    try:
        features = get_transaction_features(transaction_id)
        transaction_data = data.model_dump()

        model_features = {
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount",
        }

        compatible_features = {
            key: value
            for key, value in features.items()
            if key in model_features
        }

        transaction_data.update(compatible_features)

        result = predict_transaction(transaction_data)

        saved_result = {
            "prediction": int(result["prediction"]),
            "label": str(result["label"]),
            "fraud_probability": float(result["fraud_probability"]),
            "threshold": float(result["threshold"]),
        }

        saved_id = save_transaction(
            transaction_data,
            saved_result,
        )

        return {
            "transaction_id": transaction_id,
            "saved_transaction_id": saved_id,
            **saved_result,
            "engineered_features": features,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Prediction with retrieved features failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc


@app.post("/transactions/stream")
def stream_transaction(data: TransactionInput):
    try:
        transaction = data.model_dump()
        result = publish_transaction(transaction)

        return {
            "message": "Transaction published to Kafka successfully",
            **result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Transaction streaming failed.")

        raise HTTPException(
            status_code=500,
            detail="Transaction streaming failed. Check the server logs.",
        ) from exc


@app.get("/gnn/health")
def gnn_health():
    try:
        return get_gnn_status()

    except Exception as exc:
        logger.exception("GNN health check failed.")

        raise HTTPException(
            status_code=500,
            detail="GNN health check failed. Check the server logs.",
        ) from exc


@app.post("/gnn/predict")
def gnn_predict(request: GNNPredictionRequest):
    features = request.features
    edges = request.edges
    node_index = request.node_index

    if not features:
        raise HTTPException(
            status_code=422,
            detail="At least one node feature vector is required.",
        )

    try:
        x = np.asarray(features, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(
                "Each node must contain exactly 8 numerical features."
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "Node features must contain finite numbers."
            )

        if not isinstance(node_index, int) or not 0 <= node_index < len(x):
            raise ValueError(
                "node_index must refer to an existing node."
            )

        if edges:
            edge_array = np.asarray(edges, dtype=np.int64)

            if edge_array.ndim != 2 or edge_array.shape[1] != 2:
                raise ValueError(
                    "Each edge must contain exactly two node indices."
                )

            if np.any(edge_array < 0) or np.any(edge_array >= len(x)):
                raise ValueError(
                    "An edge references a node that does not exist."
                )

            edge_index = edge_array.T

            edge_index = np.concatenate(
                [edge_index, edge_index[::-1]],
                axis=1,
            )

        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        return predict_graph(
            features=x,
            edge_index=edge_index,
            transaction_node_index=node_index,
        )

    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("GNN inference failed.")

        raise HTTPException(
            status_code=503,
            detail="GNN prediction is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected GNN prediction error.")

        raise HTTPException(
            status_code=500,
            detail="GNN prediction failed. Check the server logs.",
        ) from exc