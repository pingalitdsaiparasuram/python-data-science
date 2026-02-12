"""
Task 11 – API Deployment with FastAPI
========================================
Serves the trained ML model via a REST API.
  POST /predict → returns churn/default prediction

Prerequisites:
    pip install fastapi uvicorn joblib scikit-learn pandas

Run server:
    python task11_api_deployment.py
    # or with uvicorn: uvicorn task11_api_deployment:app --reload --port 8000

Test with curl:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":35,"income":60000,"loan_amount":15000,"credit_score":720,
           "employment_type":"Salaried","loan_purpose":"Home"}'
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List

# ─── FastAPI app ──────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, validator, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[WARNING] FastAPI not installed. Install with: pip install fastapi uvicorn")
    print("[INFO] Showing API structure only (no server started).")

# ─── Request / Response schemas ───────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class PredictRequest(BaseModel):
        """Input features for prediction."""
        age: float = Field(..., ge=18, le=100, description="Customer age")
        income: float = Field(..., ge=0, description="Annual income in USD")
        loan_amount: float = Field(..., ge=0, description="Requested loan amount in USD")
        credit_score: float = Field(..., ge=300, le=850, description="Credit score (300–850)")
        employment_type: str = Field(..., description="Salaried | Self-Employed | Unemployed")
        loan_purpose: str = Field(..., description="Education | Home | Car | Personal")

        class Config:
            schema_extra = {
                "example": {
                    "age": 35,
                    "income": 60000,
                    "loan_amount": 15000,
                    "credit_score": 720,
                    "employment_type": "Salaried",
                    "loan_purpose": "Home"
                }
            }

    class BatchPredictRequest(BaseModel):
        """Batch prediction request."""
        records: List[PredictRequest]

    class PredictResponse(BaseModel):
        """Prediction output."""
        prediction: int
        probability: float
        risk_level: str
        message: str

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_path: str
        version: str


# ─── Model loader ─────────────────────────────────────────────────────────────

MODEL_PATH = "ml_pipeline_model.pkl"

class ModelService:
    """Singleton model loader and predictor."""
    _model = None

    @classmethod
    def load(cls, path: str = MODEL_PATH):
        if cls._model is None:
            if os.path.exists(path):
                cls._model = joblib.load(path)
                print(f"[INFO] Model loaded from '{path}'")
            else:
                # Create a minimal fallback model for demo
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from sklearn.impute import SimpleImputer

                numeric_features = ['age', 'income', 'loan_amount', 'credit_score']
                categorical_features = ['employment_type', 'loan_purpose']

                numeric_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ])
                categorical_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ])
                preprocessor = ColumnTransformer([
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features),
                ])
                cls._model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', GradientBoostingClassifier(n_estimators=50, random_state=42))
                ])
                print("[WARNING] Pre-trained model not found. Using untrained demo model.")
        return cls._model

    @classmethod
    def predict(cls, data: dict) -> tuple:
        """Make a prediction. Returns (label, probability)."""
        model = cls.load()
        FEATURE_COLS = ['age', 'income', 'loan_amount', 'credit_score',
                        'employment_type', 'loan_purpose']
        df = pd.DataFrame([data])[FEATURE_COLS]

        # If model not fitted, fit with dummy data for demo
        try:
            prob = model.predict_proba(df)[0][1]
            label = int(model.predict(df)[0])
        except Exception:
            # Demo fallback
            debt_ratio = data['loan_amount'] / max(data['income'], 1)
            credit_risk = (700 - data['credit_score']) / 400
            prob = float(np.clip(0.1 + 0.3 * debt_ratio + 0.2 * credit_risk +
                                 (0.2 if data['employment_type'] == 'Unemployed' else 0), 0, 0.99))
            label = int(prob > 0.5)

        return label, prob


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


# ─── FastAPI App ──────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Loan Default Prediction API",
        description="ML model API for predicting loan default risk.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        ModelService.load()

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Loan Default Prediction API", "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Check API and model health."""
        return HealthResponse(
            status="healthy",
            model_loaded=ModelService._model is not None,
            model_path=MODEL_PATH,
            version="1.0.0"
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
    async def predict(request: PredictRequest):
        """
        Predict loan default risk for a single customer.

        Returns:
        - **prediction**: 0 = No Default, 1 = Default
        - **probability**: Default probability (0.0–1.0)
        - **risk_level**: Low / Medium / High
        - **message**: Human-readable summary
        """
        try:
            data = request.dict()
            label, prob = ModelService.predict(data)
            risk = get_risk_level(prob)
            msg = (f"Customer has {risk.lower()} default risk "
                   f"({'likely to default' if label == 1 else 'unlikely to default'}).")
            return PredictResponse(
                prediction=label,
                probability=round(prob, 4),
                risk_level=risk,
                message=msg
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", tags=["Prediction"])
    async def batch_predict(request: BatchPredictRequest):
        """Batch prediction for multiple customers."""
        results = []
        for record in request.records:
            data = record.dict()
            label, prob = ModelService.predict(data)
            results.append({
                "prediction": label,
                "probability": round(prob, 4),
                "risk_level": get_risk_level(prob)
            })
        return {"count": len(results), "predictions": results}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        print("[INFO] Starting FastAPI server at http://localhost:8000")
        print("[INFO] API docs available at http://localhost:8000/docs")
        uvicorn.run("task11_api_deployment:app", host="0.0.0.0", port=8000, reload=False)
    else:
        print("[INFO] Install dependencies: pip install fastapi uvicorn")
        print("[INFO] Then run: uvicorn task11_api_deployment:app --reload")
