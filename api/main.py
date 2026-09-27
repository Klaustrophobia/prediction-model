from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.schemas import (
    EmpleadoInput, PrediccionOutput, BatchInput, BatchOutput,
    ModelInfo, FeatureImportanceOutput, HealthCheck,
)
from api.service import get_service
from config import TARGET_INV_MAP, ALERT_THRESHOLD, MEDIUM_THRESHOLD

app = FastAPI(
    title="API de Prediccion de Rotacion - Firmas Auditoras",
    description=(
        "API REST para predecir riesgo de rotacion de personal en firmas auditoras. "
        "Modelo XGBoost entrenado sobre datos sinteticos de KPMG, PwC y Deloitte."
    ),
    version="1.0.0",
    contact={
        "name": "Equipo ML",
        "email": "ml-team@firmaauditora.com",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "servicio": "API de Prediccion de Rotacion",
        "version": "1.0.0",
        "documentacion": "/docs",
        "endpoints": [
            "GET  /health",
            "GET  /model/info",
            "GET  /features/importance",
            "POST /predict",
            "POST /predict/batch",
        ],
    }


@app.get("/health", response_model=HealthCheck, tags=["Sistema"])
def health():
    try:
        service = get_service()
        return HealthCheck(
            status="ok",
            modelo_cargado=service.modelo is not None,
            preprocessor_cargado=service.preprocessor is not None,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        return HealthCheck(
            status=f"error: {str(e)}",
            modelo_cargado=False,
            preprocessor_cargado=False,
            timestamp=datetime.now().isoformat(),
        )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    try:
        service = get_service()
        info = service.get_info()
        return ModelInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/importance", response_model=FeatureImportanceOutput, tags=["Modelo"])
def features_importance(clase: str = "Renuncia voluntaria", top_n: int = 15):
    try:
        service = get_service()

        clase_a_idx = {v: k for k, v in TARGET_INV_MAP.items()}
        if clase not in clase_a_idx:
            raise HTTPException(
                status_code=400,
                detail=f"Clase invalida. Opciones: {list(clase_a_idx.keys())}",
            )
        idx = clase_a_idx[clase]

        top = service.importancia_global(clase_idx=idx, top_n=top_n)
        return FeatureImportanceOutput(clase=clase, top_features=top)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PrediccionOutput, tags=["Prediccion"])
def predict(empleado: EmpleadoInput):
    try:
        service = get_service()
        resultado = service.predecir_empleado(empleado.model_dump(), top_n=5)
        return PrediccionOutput(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prediccion"])
def predict_batch(batch: BatchInput):
    try:
        service = get_service()
        empleados_dict = [e.model_dump() for e in batch.empleados]
        resultados = service.predecir_batch(empleados_dict)
        return BatchOutput(total=len(resultados), predicciones=resultados)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
