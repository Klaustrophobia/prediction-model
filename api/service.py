
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (
    MODELS_DIR, TARGET_INV_MAP, TARGET_MAP,
    ALERT_THRESHOLD, MEDIUM_THRESHOLD, REPORTS_DIR,
)
from explainability.shap_analyzer import ShapAnalyzer
from alerts.alert_system import RECOMENDACIONES, RECOMENDACION_GENERICA


class PredictionService:
    """
    Servicio de prediccion que carga modelo + preprocesador y expone metodos.
    Singleton: se instancia una vez y se reutiliza en cada request.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.modelo = None
        self.preprocessor = None
        self.shap_analyzer = None
        self.feature_names = None
        self.metadata = {}
        self.info_modelo = {}

    # -----------------------------------------------------
    # Carga
    # -----------------------------------------------------
    def cargar(self, verbose: bool = True):
        """Carga modelo, preprocesador y calcula SHAP del dataset de referencia."""
        if verbose:
            print("\n" + "=" * 70)
            print("CARGANDO SERVICIO DE PREDICCION")
            print("=" * 70)

        # 1. Modelo
        ruta_modelo = MODELS_DIR / "modelo_final.pkl"
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {ruta_modelo}. "
                f"Ejecuta main.py primero para entrenar."
            )
        self.modelo = joblib.load(ruta_modelo)
        if verbose:
            print(f"  Modelo cargado: {ruta_modelo.name}")

        # 2. Preprocessor
        ruta_pp = MODELS_DIR / "preprocessor.pkl"
        if not ruta_pp.exists():
            raise FileNotFoundError(f"No se encontro el preprocesador en {ruta_pp}")
        self.preprocessor = joblib.load(ruta_pp)
        if verbose:
            print(f"  Preprocesador cargado: {ruta_pp.name}")

        # 3. Feature names
        self.feature_names = self.preprocessor.get_feature_names()

        # 4. Metadata (opcional)
        metas = sorted(MODELS_DIR.glob("metadata_*.json"))
        if metas:
            with open(metas[-1], "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            if verbose:
                print(f"  Metadata cargada: {metas[-1].name}")

        # 5. Info del modelo
        self.info_modelo = {
            "nombre": self.metadata.get("nombre_modelo", "XGBoost"),
            "version": "1.0.0",
            "tipo": "Gradient Boosting Classifier",
            "metricas": self.metadata.get("metricas_test", {}),
            "features": len(self.feature_names),
            "clases": [TARGET_INV_MAP[i] for i in range(3)],
            "fecha_entrenamiento": self.metadata.get("timestamp"),
        }

        # 6. Calcular SHAP sobre el dataset de referencia
        #    Usamos el Excel original (los 600) como referencia.
        if verbose:
            print("  Calculando SHAP sobre el dataset de referencia...")

        from data_pipeline.data_loader import cargar_datos
        df_ref = cargar_datos(verbose=False)
        X_ref = self.preprocessor.transform(df_ref)
        self.shap_analyzer = ShapAnalyzer(self.modelo, self.feature_names)
        self.shap_analyzer.calcular(X_ref, verbose=False)

        if verbose:
            print("  Servicio listo.")
            print("=" * 70)

    # -----------------------------------------------------
    # Prediccion individual
    # -----------------------------------------------------
    def predecir_empleado(self, datos: dict, top_n: int = 5) -> dict:
        """
        Predice riesgo de un empleado.
        Args:
            datos: dict con las features del empleado
            top_n: numero de factores SHAP a mostrar
        Returns:
            dict con prediccion + factores + recomendaciones
        """
        # Convertir a DataFrame de 1 fila
        df_input = pd.DataFrame([datos])

        # Asegurar que existan todas las columnas requeridas
        df_input = self._completar_columnas(df_input)

        # Preprocesar
        X_proc = self.preprocessor.transform(df_input)

        # Predecir
        proba = self.modelo.predict_proba(X_proc)[0]
        pred_clase = int(np.argmax(proba))

        # Nivel de riesgo (foco en renuncia voluntaria)
        p_vol = float(proba[1])
        if p_vol >= ALERT_THRESHOLD:
            nivel = "Alto"
        elif p_vol >= MEDIUM_THRESHOLD:
            nivel = "Moderado"
        else:
            nivel = "Bajo"

        # Explicabilidad SHAP
        factores_riesgo = []
        factores_proteccion = []
        recomendaciones = []

        try:
            # Calcular SHAP para esta fila
            import shap
            explainer = shap.TreeExplainer(self.modelo)
            shap_values = explainer.shap_values(X_proc)

            # Normalizar a lista de matrices
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

            # SHAP de la clase 1 (Renuncia voluntaria)
            valores_vol = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

            pares = list(zip(self.feature_names, valores_vol, X_proc[0]))
            ordenados = sorted(pares, key=lambda t: abs(t[1]), reverse=True)[:top_n]

            for feat, val_shap, val_feat in ordenados:
                item = {
                    "feature": feat,
                    "shap": float(val_shap),
                    "valor": float(val_feat) if not isinstance(val_feat, str) else 0.0,
                }
                if val_shap > 0:
                    factores_riesgo.append(item)
                    rec = RECOMENDACIONES.get(feat, RECOMENDACION_GENERICA)
                    recomendaciones.append(rec)
                else:
                    factores_proteccion.append(item)

        except Exception as e:
            # Si SHAP falla, no bloqueamos la prediccion
            factores_riesgo = []
            factores_proteccion = []
            recomendaciones = ["No se pudieron generar recomendaciones automaticas"]

        return {
            "ID_Empleado": datos.get("ID_Empleado"),
            "probabilidad_permanencia": float(proba[0]),
            "probabilidad_renuncia_voluntaria": float(proba[1]),
            "probabilidad_renuncia_involuntaria": float(proba[2]),
            "nivel_riesgo": nivel,
            "prediccion": TARGET_INV_MAP[pred_clase],
            "factores_riesgo": factores_riesgo,
            "factores_proteccion": factores_proteccion,
            "recomendaciones": recomendaciones,
        }

    # -----------------------------------------------------
    # Prediccion masiva
    # -----------------------------------------------------
    def predecir_batch(self, empleados: list) -> list:
        """Predice multiples empleados."""
        resultados = []
        for emp in empleados:
            try:
                r = self.predecir_empleado(emp, top_n=3)
                resultados.append(r)
            except Exception as e:
                resultados.append({
                    "ID_Empleado": emp.get("ID_Empleado", "unknown"),
                    "error": str(e),
                })
        return resultados

    # -----------------------------------------------------
    # Importancia global de features
    # -----------------------------------------------------
    def importancia_global(self, clase_idx: int = 1, top_n: int = 15) -> list:
        """Retorna top N features mas importantes por SHAP para una clase."""
        df = self.shap_analyzer.importancia_global(clase_idx).head(top_n)
        return [
            {
                "rank": int(i + 1),
                "feature": str(row["Feature"]),
                "importancia": float(row["Importancia_SHAP"]),
            }
            for i, row in df.iterrows()
        ]

    # -----------------------------------------------------
    # Completar columnas faltantes (por si el JSON de entrada es parcial)
    # -----------------------------------------------------
    def _completar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegura que el DataFrame tenga todas las columnas que espera el preprocessor."""
        # Columnas que espera el preprocessor (originales, antes de FE)
        from config import NUM_FEATURES, CAT_FEATURES
        requeridas = NUM_FEATURES + CAT_FEATURES

        for col in requeridas:
            if col not in df.columns:
                # Rellenar con valor por defecto
                if col in CAT_FEATURES:
                    df[col] = "Desconocido"
                else:
                    df[col] = 0

        return df

    # -----------------------------------------------------
    # Info del modelo
    # -----------------------------------------------------
    def get_info(self) -> dict:
        return self.info_modelo


# Instancia singleton
_service = None


def get_service() -> PredictionService:
    """Retorna la instancia unica del servicio, cargandola si es necesario."""
    global _service
    if _service is None:
        _service = PredictionService()
        _service.cargar()
    return _service