import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, TARGET_INV_MAP


class BaselineModelSelector:

    def __init__(self):
        self.models = self._definir_modelos()
        self.resultados = {}

    def _definir_modelos(self) -> dict:
        """Define el diccionario de modelos a evaluar."""
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        }

    def _calcular_metricas(self, y_true, y_pred, y_proba) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc_macro": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        }

    def entrenar_todos(self, X_train, y_train, X_val, y_val, verbose=True):
        """
        """
        print("\n" + "=" * 78)
        print("ENTRENAMIENTO DE MODELOS BASELINE")
        print("=" * 78)
        print(f"  Train: {X_train.shape[0]} registros, {X_train.shape[1]} features")
        print(f"  Val:   {X_val.shape[0]} registros\n")

        for nombre, modelo in self.models.items():
            print("-" * 78)
            print(f"  Modelo: {nombre}")
            print("-" * 78)

            t0 = time.time()
            modelo.fit(X_train, y_train)
            t_fit = time.time() - t0

            y_pred = modelo.predict(X_val)
            y_proba = modelo.predict_proba(X_val)

            metricas = self._calcular_metricas(y_val, y_pred, y_proba)
            metricas["tiempo_fit_seg"] = t_fit

            self.resultados[nombre] = {
                "modelo": modelo,
                "metricas": metricas,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }

            if verbose:
                print(f"  Tiempo de entrenamiento: {t_fit:.2f} s")
                print(f"  Accuracy:         {metricas['accuracy']:.4f}")
                print(f"  Precision (mac):  {metricas['precision_macro']:.4f}")
                print(f"  Recall (macro):   {metricas['recall_macro']:.4f}")
                print(f"  F1 (macro):       {metricas['f1_macro']:.4f}")
                print(f"  F1 (weighted):    {metricas['f1_weighted']:.4f}")
                print(f"  ROC-AUC (macro):  {metricas['roc_auc_macro']:.4f}")
                print()

        return self.resultados

    def tabla_comparativa(self) -> "pd.DataFrame":
        import pandas as pd

        filas = []
        for nombre, data in self.resultados.items():
            m = data["metricas"]
            filas.append({
                "Modelo": nombre,
                "Accuracy": round(m["accuracy"], 4),
                "Precision_macro": round(m["precision_macro"], 4),
                "Recall_macro": round(m["recall_macro"], 4),
                "F1_macro": round(m["f1_macro"], 4),
                "F1_weighted": round(m["f1_weighted"], 4),
                "ROC_AUC_macro": round(m["roc_auc_macro"], 4),
                "Tiempo_seg": round(m["tiempo_fit_seg"], 2),
            })

        df = pd.DataFrame(filas).sort_values("ROC_AUC_macro", ascending=False).reset_index(drop=True)
        return df

    def mejor_modelo(self) -> tuple:
        """Devuelve (nombre, modelo) del mejor segun ROC-AUC macro."""
        mejor = max(self.resultados.items(), key=lambda x: x[1]["metricas"]["roc_auc_macro"])
        return mejor[0], mejor[1]["modelo"]

    def imprimir_matriz_confusion(self, nombre: str, y_val):
        """Imprime la matriz de confusion de un modelo."""
        y_pred = self.resultados[nombre]["y_pred"]
        cm = confusion_matrix(y_val, y_pred)
        clases = [TARGET_INV_MAP[i] for i in range(len(TARGET_INV_MAP))]

        print(f"\n  Matriz de confusion - {nombre}")
        print("  " + "-" * 70)
        header = "  " + " " * 30 + "  ".join(f"{c[:12]:>12}" for c in clases)
        print(header)
        for i, fila in enumerate(cm):
            valores = "  ".join(f"{v:>12}" for v in fila)
            print(f"  {clases[i][:28]:<30}{valores}")

    def imprimir_reporte_clasificacion(self, nombre: str, y_val):
        y_pred = self.resultados[nombre]["y_pred"]
        clases = [TARGET_INV_MAP[i] for i in range(len(TARGET_INV_MAP))]

        print(f"\n  Reporte de clasificacion - {nombre}")
        print("  " + "-" * 70)
        print(classification_report(y_val, y_pred, target_names=clases, digits=4, zero_division=0))


if __name__ == "__main__":
    from data_pipeline.data_loader import cargar_datos
    from data_pipeline.preprocessing import preparar_datos

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, pp = preparar_datos(df, verbose=False)

    selector = BaselineModelSelector()
    selector.entrenar_todos(X_train, y_train, X_val, y_val)
    print(selector.tabla_comparativa().to_string(index=False))