from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys
from pathlib import Path

# Asegurar imports desde raiz
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.schemas import (
    EmpleadoInput, PrediccionOutput, BatchInput, BatchOutput,
    ModelInfo, FeatureImportanceOutput, HealthCheck,
)
from api.service import get_service
from config import TARGET_INV_MAP, ALERT_THRESHOLD, MEDIUM_THRESHOLD


# ============================================================
# App + CORS
# ============================================================
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

# CORS para que el dashboard (Streamlit) pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En produccion, restringir a dominios especificos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["Root"])
def root():
    """Endpoint raiz con informacion basica."""
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
    """Verifica que el servicio este activo y el modelo cargado."""
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
    """Retorna informacion sobre el modelo entrenado."""
    try:
        service = get_service()
        info = service.get_info()
        return ModelInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/importance", response_model=FeatureImportanceOutput, tags=["Modelo"])
def features_importance(clase: str = "Renuncia voluntaria", top_n: int = 15):
    """
    Retorna las top N features mas importantes segun SHAP.
    Args:
        clase: 'Permanencia', 'Renuncia voluntaria' o 'Renuncia involuntaria'
        top_n: numero de features a retornar
    """
    try:
        service = get_service()

        # Mapear clase a indice
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
    """
    Predice el riesgo de rotacion de un empleado.
    Retorna probabilidades, nivel de riesgo, factores SHAP y recomendaciones.
    """
    try:
        service = get_service()
        resultado = service.predecir_empleado(empleado.model_dump(), top_n=5)
        return PrediccionOutput(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prediccion"])
def predict_batch(batch: BatchInput):
    """
    Predice el riesgo de multiples empleados.
    Util para procesar la plantilla completa.
    """
    try:
        service = get_service()
        empleados_dict = [e.model_dump() for e in batch.empleados]
        resultados = service.predecir_batch(empleados_dict)
        return BatchOutput(total=len(resultados), predicciones=resultados)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Ejecucion directa
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)