import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score,
)

from config import FIGURES_DIR, REPORTS_DIR, TARGET_MAP


# ============================================================
# METRICAS DE FAIRNESS (implementacion directa, sin fairlearn)
# ============================================================
def demographic_parity_difference(y_pred, sensitive) -> float:
    """
    Diferencia entre la tasa de prediccion positiva del grupo
    mas favorecido y el menos favorecido.
    """
    grupos = pd.unique(sensitive)
    tasas = {}
    for g in grupos:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        tasas[g] = y_pred[mask].mean()
    if len(tasas) < 2:
        return 0.0
    return float(max(tasas.values()) - min(tasas.values()))


def disparate_impact_ratio(y_pred, sensitive) -> float:
    """
    Ratio entre la tasa de prediccion positiva del grupo menos favorecido
    y la del grupo mas favorecido. Valor ideal: cercano a 1.
    Regla 4/5 de EEOC: debe estar entre 0.8 y 1.25.
    """
    grupos = pd.unique(sensitive)
    tasas = {}
    for g in grupos:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        tasas[g] = y_pred[mask].mean()
    if len(tasas) < 2:
        return 1.0
    max_tasa = max(tasas.values())
    min_tasa = min(tasas.values())
    if max_tasa == 0:
        return 1.0
    return float(min_tasa / max_tasa)


def equalized_odds_difference(y_true, y_pred, sensitive) -> dict:
    """
    Diferencia en TPR y FPR entre grupos.
    Retorna TPR_diff y FPR_diff por subgrupo.
    """
    grupos = pd.unique(sensitive)
    tprs = {}
    fprs = {}
    for g in grupos:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs[g] = tpr
        fprs[g] = fpr

    if len(tprs) < 2:
        return {"tpr_difference": 0.0, "fpr_difference": 0.0,
                "tprs": tprs, "fprs": fprs}

    return {
        "tpr_difference": float(max(tprs.values()) - min(tprs.values())),
        "fpr_difference": float(max(fprs.values()) - min(fprs.values())),
        "tprs": tprs,
        "fprs": fprs,
    }


# ============================================================
# CLASE PRINCIPAL
# ============================================================
class BiasAnalyzer:
    """
    Analiza fairness del modelo sobre variables sensibles.
    Binzariza el target (Permanencia=0 vs Cualquier renuncia=1) para el analisis.
    """

    def __init__(self, modelo, preprocessor):
        self.modelo = modelo
        self.preprocessor = preprocessor
        self.resultados_ = {}

    # -----------------------------------------------------
    # Preparar datos binarios
    # -----------------------------------------------------
    def _preparar_binario(self, df_empleados: pd.DataFrame):
        """
        Extrae:
          - y_true binario (1 = cualquier renuncia, 0 = permanencia)
          - y_pred binario (1 = predicho como renuncia voluntaria o involuntaria)
          - y_proba (probabilidad de renuncia voluntaria)
          - Dataframe con variables sensibles
        """
        if "Rotacion_Personal" not in df_empleados.columns:
            raise ValueError("La columna objetivo no esta en el DataFrame.")

        # y_true binario
        y_true_bin = (df_empleados["Rotacion_Personal"] != "Permanencia").astype(int).values

        # Predecir
        X_proc = self.preprocessor.transform(df_empleados)
        probas = self.modelo.predict_proba(X_proc)

        # Prediccion binaria: es renuncia si la clase dominante no es Permanencia
        y_pred_multiclase = np.argmax(probas, axis=1)
        y_pred_bin = (y_pred_multiclase != 0).astype(int)

        # Dataframe de sensibles
        sensibles = pd.DataFrame(index=df_empleados.index)

        if "Genero" in df_empleados.columns:
            sensibles["Genero"] = df_empleados["Genero"].values

        if "Edad" in df_empleados.columns:
            edades = df_empleados["Edad"].values
            bins = [0, 25, 30, 35, 40, 100]
            labels = ["21-25", "26-30", "31-35", "36-40", "41+"]
            sensibles["Grupo_Edad"] = pd.cut(edades, bins=bins, labels=labels).astype(str)

        if "Firma_Auditora" in df_empleados.columns:
            sensibles["Firma_Auditora"] = df_empleados["Firma_Auditora"].values

        return y_true_bin, y_pred_bin, probas, sensibles

    # -----------------------------------------------------
    # Analisis de una variable sensible
    # -----------------------------------------------------
    def analizar_sensible(self, y_true, y_pred, sensitive_series, nombre_sensible):
        """Analiza fairness para una variable sensible."""
        sensitive = sensitive_series.values
        grupos = pd.unique(sensitive)

        # Metricas de fairness globales
        dp = demographic_parity_difference(y_pred, sensitive)
        di = disparate_impact_ratio(y_pred, sensitive)
        eo = equalized_odds_difference(y_true, y_pred, sensitive)

        # Metricas por grupo
        metricas_grupo = {}
        for g in grupos:
            mask = sensitive == g
            if mask.sum() == 0:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]

            if len(np.unique(yt)) > 1:
                prec = precision_score(yt, yp, zero_division=0)
                rec = recall_score(yt, yp, zero_division=0)
                f1 = f1_score(yt, yp, zero_division=0)
            else:
                prec = rec = f1 = 0.0

            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            metricas_grupo[str(g)] = {
                "n": int(mask.sum()),
                "tasa_prediccion_positiva": float(yp.mean()),
                "precision": float(prec),
                "recall_tpr": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
            }

        return {
            "variable_sensible": nombre_sensible,
            "demographic_parity_diff": float(dp),
            "disparate_impact_ratio": float(di),
            "tpr_difference": eo["tpr_difference"],
            "fpr_difference": eo["fpr_difference"],
            "metricas_por_grupo": metricas_grupo,
            "cumple_regla_4_5": bool(0.8 <= di <= 1.25),
            "cumple_dp_umbral": bool(dp < 0.10),
            "cumple_eo_umbral": bool(eo["tpr_difference"] < 0.10 and eo["fpr_difference"] < 0.10),
        }

    # -----------------------------------------------------
    # Analizar todas las variables sensibles
    # -----------------------------------------------------
    def analizar_todas(self, df_empleados: pd.DataFrame, verbose: bool = True) -> dict:
        """Ejecuta el analisis de fairness sobre todas las variables sensibles."""
        if verbose:
            print("\n  Preparando datos para analisis de sesgos...")

        y_true, y_pred, probas, sensibles = self._preparar_binario(df_empleados)

        # Analizar cada variable sensible disponible
        for col in sensibles.columns:
            if verbose:
                print(f"\n  Analizando: {col}")
            self.resultados_[col] = self.analizar_sensible(
                y_true, y_pred, sensibles[col], col
            )

        if verbose:
            self._imprimir_consola()

        return self.resultados_

    # -----------------------------------------------------
    # Impresion en consola
    # -----------------------------------------------------
    def _imprimir_consola(self):
        print("\n" + "=" * 78)
        print("PASO 8: ANALISIS DE SESGOS Y FAIRNESS")
        print("=" * 78)

        for nombre, r in self.resultados_.items():
            print(f"\n  Variable sensible: {nombre}")
            print("  " + "-" * 70)
            print(f"    Demographic Parity Diff:     {r['demographic_parity_diff']:.4f} "
                  f"({'OK' if r['cumple_dp_umbral'] else 'REVISAR'})")
            print(f"    Disparate Impact Ratio:      {r['disparate_impact_ratio']:.4f} "
                  f"({'OK' if r['cumple_regla_4_5'] else 'REVISAR'})")
            print(f"    TPR Difference (Equalized):  {r['tpr_difference']:.4f}")
            print(f"    FPR Difference (Equalized):  {r['fpr_difference']:.4f} "
                  f"({'OK' if r['cumple_eo_umbral'] else 'REVISAR'})")

            print(f"\n    Detalle por grupo:")
            print(f"    {'Grupo':<20}{'n':>6}{'Tasa Pos':>12}{'Precision':>12}"
                  f"{'Recall':>12}{'F1':>10}{'FPR':>10}")
            print("    " + "-" * 84)
            for grupo, m in sorted(r["metricas_por_grupo"].items()):
                print(f"    {grupo[:18]:<20}{m['n']:>6}"
                      f"{m['tasa_prediccion_positiva']:>12.4f}"
                      f"{m['precision']:>12.4f}"
                      f"{m['recall_tpr']:>12.4f}"
                      f"{m['f1']:>10.4f}"
                      f"{m['fpr']:>10.4f}")

    # -----------------------------------------------------
    # Graficos
    # -----------------------------------------------------
    def plot_fairness_bars(self, nombre_sensible: str) -> Path:
        """Grafico comparativo de metricas por grupo para una variable sensible."""
        if nombre_sensible not in self.resultados_:
            raise ValueError(f"No hay analisis para: {nombre_sensible}")

        r = self.resultados_[nombre_sensible]
        grupos = list(r["metricas_por_grupo"].keys())
        tasas = [r["metricas_por_grupo"][g]["tasa_prediccion_positiva"] for g in grupos]
        recalls = [r["metricas_por_grupo"][g]["recall_tpr"] for g in grupos]
        fprs = [r["metricas_por_grupo"][g]["fpr"] for g in grupos]

        x = np.arange(len(grupos))
        width = 0.27

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, tasas, width, label="Tasa prediccion positiva", color="#3498db")
        ax.bar(x, recalls, width, label="Recall (TPR)", color="#2ecc71")
        ax.bar(x + width, fprs, width, label="FPR", color="#e74c3c")

        ax.set_xticks(x)
        ax.set_xticklabels(grupos, rotation=20)
        ax.set_ylabel("Valor")
        ax.set_title(f"Fairness por {nombre_sensible}", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Linea de referencia 0.5
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)

        plt.tight_layout()
        ruta = FIGURES_DIR / f"10_fairness_{nombre_sensible}.png"
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        return ruta

    # -----------------------------------------------------
    # Guardar reporte
    # -----------------------------------------------------
    def guardar_reporte(self, ruta: Path = None) -> Path:
        """Guarda el analisis de sesgos en un archivo .txt."""
        if ruta is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta = REPORTS_DIR / f"fairness_{timestamp}.txt"

        lineas = []
        lineas.append("=" * 78)
        lineas.append("REPORTE DE FAIRNESS Y SESGOS")
        lineas.append("=" * 78)
        lineas.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lineas.append("")
        lineas.append("Metricas calculadas sobre el dataset completo (600 empleados).")
        lineas.append("Target binarizado: 0 = Permanencia, 1 = Cualquier renuncia.")
        lineas.append("")

        for nombre, r in self.resultados_.items():
            lineas.append("-" * 78)
            lineas.append(f"Variable sensible: {nombre}")
            lineas.append("-" * 78)
            lineas.append(f"  Demographic Parity Diff:     {r['demographic_parity_diff']:.4f}")
            lineas.append(f"  Disparate Impact Ratio:      {r['disparate_impact_ratio']:.4f}")
            lineas.append(f"  TPR Difference:              {r['tpr_difference']:.4f}")
            lineas.append(f"  FPR Difference:              {r['fpr_difference']:.4f}")
            lineas.append(f"  Cumple regla 4/5 (EEOC):     {r['cumple_regla_4_5']}")
            lineas.append(f"  Cumple umbral DP < 0.10:     {r['cumple_dp_umbral']}")
            lineas.append(f"  Cumple umbral EO < 0.10:     {r['cumple_eo_umbral']}")
            lineas.append("")
            lineas.append("  Detalle por grupo:")
            lineas.append(f"  {'Grupo':<22}{'n':>6}{'Tasa Pos':>12}"
                          f"{'Precision':>12}{'Recall':>12}{'F1':>10}{'FPR':>10}")
            lineas.append("  " + "-" * 84)
            for grupo, m in sorted(r["metricas_por_grupo"].items()):
                lineas.append(f"  {grupo[:20]:<22}{m['n']:>6}"
                              f"{m['tasa_prediccion_positiva']:>12.4f}"
                              f"{m['precision']:>12.4f}"
                              f"{m['recall_tpr']:>12.4f}"
                              f"{m['f1']:>10.4f}"
                              f"{m['fpr']:>10.4f}")
            lineas.append("")

        lineas.append("=" * 78)
        lineas.append("INTERPRETACION")
        lineas.append("=" * 78)
        lineas.append("  - Demographic Parity Diff < 0.10: paridad entre grupos")
        lineas.append("  - Disparate Impact Ratio entre 0.8 y 1.25: regla 4/5 EEOC")
        lineas.append("  - Equalized Odds Diff < 0.10: mismo TPR/FPR entre grupos")
        lineas.append("  - Valores fuera de rango indican posible sesgo a revisar")
        lineas.append("=" * 78)

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))

        print(f"\n  Reporte de fairness guardado en: {ruta}")
        return ruta


# ============================================================
# FUNCION DE ALTO NIVEL
# ============================================================
def ejecutar_analisis_sesgos(modelo, preprocessor, df_empleados: pd.DataFrame,
                              verbose: bool = True):
    """
    Ejecuta el PASO 8 completo:
      1. Analiza fairness por Genero, Grupo_Edad y Firma
      2. Genera graficos comparativos
      3. Guarda reporte .txt
    Retorna el BiasAnalyzer.
    """
    print("\n" + "=" * 78)
    print("PASO 8: ETICA Y ANALISIS DE SESGOS")
    print("=" * 78)

    analyzer = BiasAnalyzer(modelo, preprocessor)
    analyzer.analizar_todas(df_empleados, verbose=verbose)

    print("\n  Generando graficos de fairness...")
    print("  " + "-" * 70)
    for nombre in analyzer.resultados_.keys():
        try:
            r = analyzer.plot_fairness_bars(nombre)
            print(f"    {r.name}")
        except Exception as e:
            print(f"    [SKIP] {nombre}: {e}")

    analyzer.guardar_reporte()

    return analyzer


if __name__ == "__main__":
    import joblib
    from config import MODELS_DIR
    from data_pipeline.data_loader import cargar_datos

    df = cargar_datos()
    modelo = joblib.load(MODELS_DIR / "modelo_final.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

    ejecutar_analisis_sesgos(modelo, preprocessor, df)