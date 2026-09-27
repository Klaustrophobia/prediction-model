import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from config import MODELS_DIR, RANDOM_STATE, TARGET_INV_MAP
from modeling.optimizer import optimizar_modelo, evaluar_en_test


class ModelTrainer:


    def __init__(self, nombre_modelo: str = "XGBoost", n_trials: int = 30):
        self.nombre_modelo = nombre_modelo
        self.n_trials = n_trials
        self.modelo_ = None
        self.params_ = None
        self.score_cv_ = None
        self.metricas_test_ = None

    def entrenar(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        verbose: bool = True,
    ):
        print("\n" + "=" * 78)
        print(f"PASO 5: OPTIMIZACION DE HIPERPARAMETROS - {self.nombre_modelo}")
        print("=" * 78)

        self.modelo_, self.params_, self.score_cv_ = optimizar_modelo(
            self.nombre_modelo, X_train, y_train,
            n_trials=self.n_trials, verbose=verbose,
        )
        return self

    def evaluar(self, X_test, y_test):
        self.metricas_test_, y_pred, y_proba = evaluar_en_test(
            self.modelo_, X_test, y_test, f"{self.nombre_modelo} optimizado"
        )
        return self.metricas_test_

    def guardar(self):
        if self.modelo_ is None:
            raise ValueError("No hay modelo entrenado. Llama a entrenar() primero.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"modelo_{self.nombre_modelo.lower()}_{timestamp}.pkl"
        ruta_modelo = MODELS_DIR / nombre_archivo
        joblib.dump(self.modelo_, ruta_modelo)
        print(f"\n  Modelo guardado en: {ruta_modelo}")

        ruta_final = MODELS_DIR / "modelo_final.pkl"
        joblib.dump(self.modelo_, ruta_final)
        print(f"  Copia como modelo final: {ruta_final}")

        metadatos = {
            "nombre_modelo": self.nombre_modelo,
            "timestamp": timestamp,
            "score_cv": self.score_cv_,
            "params": self.params_,
            "metricas_test": self.metricas_test_,
        }
        ruta_meta = MODELS_DIR / f"metadata_{self.nombre_modelo.lower()}_{timestamp}.json"
        with open(ruta_meta, "w", encoding="utf-8") as f:
            json.dump(metadatos, f, indent=2, default=str)
        print(f"  Metadatos guardados en: {ruta_meta}")

        return ruta_modelo


if __name__ == "__main__":
    from data_pipeline.data_loader import cargar_datos
    from data_pipeline.preprocessing import preparar_datos

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preparar_datos(df, verbose=False)

    trainer = ModelTrainer(nombre_modelo="XGBoost", n_trials=10)
    trainer.entrenar(X_train, y_train, verbose=True)
    trainer.evaluar(X_test, y_test)
    trainer.guardar()