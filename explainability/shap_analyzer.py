import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI (evita problemas en Windows headless)
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from config import FIGURES_DIR, TARGET_INV_MAP


class ShapAnalyzer:
    """
    Encapsula el calculo y visualizacion de SHAP values.
    """

    def __init__(self, modelo, feature_names, class_names=None):
        """
        Args:
            modelo: modelo entrenado (XGBoost, LightGBM, etc.)
            feature_names: lista con los nombres de las features
            class_names: lista con los nombres de las clases
        """
        self.modelo = modelo
        self.feature_names = list(feature_names)
        self.class_names = class_names or [TARGET_INV_MAP[i] for i in range(3)]
        self.explainer = None
        self.shap_values = None

    # -----------------------------------------------------
    # 1. Calcular SHAP values
    # -----------------------------------------------------
    def calcular(self, X, verbose: bool = True):
        """
        Calcula SHAP values con TreeExplainer.
        Retorna la lista de matrices (una por clase).
        """
        if verbose:
            print(f"\n  Calculando SHAP values sobre {X.shape[0]} registros...")

        self.explainer = shap.TreeExplainer(self.modelo)
        shap_values = self.explainer.shap_values(X)

        # Normalizar a lista de matrices (una por clase)
        # Algunas versiones de XGBoost + SHAP devuelven:
        #   - lista de arrays (lo esperado)
        #   - array 3D (n_samples, n_features, n_classes)
        #   - lista de listas (raras)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif isinstance(shap_values, list) and len(shap_values) > 0:
            # Si es lista pero cada elemento tiene dimensiones raras, normalizar
            shap_values = [np.asarray(sv) for sv in shap_values]

        self.shap_values = shap_values

        if verbose:
            n_clases = len(shap_values)
            n_features = shap_values[0].shape[1]
            print(f"  SHAP calculado: {n_clases} clases x {n_features} features")

        return shap_values

    # -----------------------------------------------------
    # 2. Importancia global (mean |SHAP|)
    # -----------------------------------------------------
    def importancia_global(self, clase_idx: int = 1) -> pd.DataFrame:
        """
        Devuelve un DataFrame con la importancia media absoluta por feature.
        Por defecto usa la clase 1 (Renuncia voluntaria).
        """
        if self.shap_values is None:
            raise ValueError("Primero llama a calcular().")

        valores = np.abs(self.shap_values[clase_idx]).mean(axis=0)
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importancia_SHAP": valores,
        }).sort_values("Importancia_SHAP", ascending=False).reset_index(drop=True)

        return df

    # -----------------------------------------------------
    # 3. Grafico summary (beeswarm)
    # -----------------------------------------------------
    def plot_summary(self, X, clase_idx: int = 1, max_display: int = 20) -> Path:
        """
        Genera el summary plot (beeswarm) para una clase.
        """
        plt.figure(figsize=(11, 8))
        shap.summary_plot(
            self.shap_values[clase_idx],
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(
            f"SHAP Summary - Clase: {self.class_names[clase_idx]}",
            fontweight="bold", fontsize=12,
        )
        plt.tight_layout()

        nombre = f"06_shap_summary_clase_{clase_idx}.png"
        ruta = FIGURES_DIR / nombre
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        return ruta

    # -----------------------------------------------------
    # 4. Grafico bar (importancia media)
    # -----------------------------------------------------
    def plot_importancia_bar(self, clase_idx: int = 1, top_n: int = 20) -> Path:
        """
        Genera un grafico de barras con la importancia media |SHAP|.
        No requiere X porque usa self.shap_values internamente.
        """
        df = self.importancia_global(clase_idx).head(top_n)

        plt.figure(figsize=(10, max(5, top_n * 0.35)))
        plt.barh(df["Feature"][::-1], df["Importancia_SHAP"][::-1],
                 color="#3498db", edgecolor="black", linewidth=0.5)
        plt.xlabel("Importancia media |SHAP|", fontsize=10)
        plt.title(
            f"Top {top_n} Features - Clase: {self.class_names[clase_idx]}",
            fontweight="bold", fontsize=12,
        )
        plt.tight_layout()

        nombre = f"07_shap_importancia_clase_{clase_idx}.png"
        ruta = FIGURES_DIR / nombre
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        return ruta

    # -----------------------------------------------------
    # 5. Dependencia de una feature especifica
    # -----------------------------------------------------
    def plot_dependence(self, feature_name: str, X, clase_idx: int = 1) -> Path:
        """
        Genera un dependence plot para una feature especifica.
        Requiere X (valores reales) para colorear por interacciones.
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature no encontrada: {feature_name}")

        idx = self.feature_names.index(feature_name)

        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            idx,
            self.shap_values[clase_idx],
            X,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title(
            f"SHAP Dependence - {feature_name} ({self.class_names[clase_idx]})",
            fontweight="bold", fontsize=11,
        )
        plt.tight_layout()

        # Limpiar nombre para archivo
        nombre_limpio = (feature_name
                         .replace("/", "_")
                         .replace("\\", "_")
                         .replace(" ", "_"))
        nombre = f"08_shap_dependence_{nombre_limpio}.png"
        ruta = FIGURES_DIR / nombre
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        return ruta

    # -----------------------------------------------------
    # 6. Explicacion individual (waterfall)
    # -----------------------------------------------------
    def plot_waterfall_individual(self, X, idx_registro: int, clase_idx: int = 1,
                                  max_display: int = 12) -> Path:
        """
        Genera un waterfall plot para un registro individual.
        """
        # Obtener base value de forma robusta
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[clase_idx]
        else:
            base_value = self.explainer.expected_value

        plt.figure(figsize=(10, 7))
        explicacion = shap.Explanation(
            values=self.shap_values[clase_idx][idx_registro],
            base_values=base_value,
            data=X[idx_registro],
            feature_names=self.feature_names,
        )
        shap.waterfall_plot(explicacion, max_display=max_display, show=False)
        plt.title(
            f"SHAP Waterfall - Registro #{idx_registro} ({self.class_names[clase_idx]})",
            fontweight="bold", fontsize=11,
        )
        plt.tight_layout()

        nombre = f"09_shap_waterfall_reg{idx_registro}_clase{clase_idx}.png"
        ruta = FIGURES_DIR / nombre
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        return ruta

    # -----------------------------------------------------
    # 7. Top factores por registro (tabla)
    # -----------------------------------------------------
    def top_factores_individual(self, X, idx_registro: int, clase_idx: int = 1,
                                 n: int = 5) -> dict:
        """
        Devuelve los top N factores que empujan hacia la clase dada
        para un registro especifico.
        """
        valores = self.shap_values[clase_idx][idx_registro]
        datos = X[idx_registro]

        pares = list(zip(self.feature_names, valores, datos))
        ordenados = sorted(pares, key=lambda t: abs(t[1]), reverse=True)[:n]

        empuja_hacia = []
        empuja_contra = []
        for feat, val_shap, val_feat in ordenados:
            item = {
                "feature": feat,
                "shap": float(val_shap),
                "valor": float(val_feat) if not isinstance(val_feat, str) else val_feat,
            }
            if val_shap > 0:
                empuja_hacia.append(item)
            else:
                empuja_contra.append(item)

        return {
            "empuja_hacia": empuja_hacia,
            "empuja_contra": empuja_contra,
        }

    # -----------------------------------------------------
    # 8. Reporte completo en consola
    # -----------------------------------------------------
    def reporte_consola(self, X, clase_idx: int = 1, top_n: int = 15):
        """Imprime un reporte completo en consola."""
        print("\n" + "=" * 78)
        print(f"REPORTE SHAP - CLASE: {self.class_names[clase_idx]}")
        print("=" * 78)

        df_imp = self.importancia_global(clase_idx).head(top_n)
        print(f"\n  Top {top_n} features mas importantes (|SHAP| medio):")
        print("  " + "-" * 70)
        print(f"  {'Rank':<6}{'Feature':<38}{'Importancia':>15}")
        print("  " + "-" * 70)
        for i, row in df_imp.iterrows():
            print(f"  {i+1:<6}{row['Feature']:<38}{row['Importancia_SHAP']:>15.5f}")

        # Explicar registro 0 como ejemplo
        print(f"\n  Ejemplo de explicacion individual (registro #0):")
        print("  " + "-" * 70)
        factores = self.top_factores_individual(X, 0, clase_idx, n=5)

        print(f"  Factores que EMPUJAN HACIA '{self.class_names[clase_idx]}':")
        if factores["empuja_hacia"]:
            for f in factores["empuja_hacia"]:
                print(f"    + {f['feature']:<38} SHAP = {f['shap']:+.4f}")
        else:
            print("    (ninguno)")

        print(f"\n  Factores que EMPUJAN EN CONTRA:")
        if factores["empuja_contra"]:
            for f in factores["empuja_contra"]:
                print(f"    - {f['feature']:<38} SHAP = {f['shap']:+.4f}")
        else:
            print("    (ninguno)")


# ============================================================
# FUNCION DE ALTO NIVEL PARA USAR EN main.py
# ============================================================
def ejecutar_shap(modelo, X_train, X_test, feature_names, verbose: bool = True):
    """
    Ejecuta el analisis SHAP completo:
      1. Calcula SHAP values sobre test
      2. Importancia global de las 3 clases
      3. Graficos summary, bar, waterfall y dependence
      4. Reporte en consola
    Retorna (analyzer, idx_max_riesgo, proba_voluntaria).
    """
    print("\n" + "=" * 78)
    print("PASO 6: EXPLICABILIDAD CON SHAP")
    print("=" * 78)

    analyzer = ShapAnalyzer(modelo, feature_names)
    analyzer.calcular(X_test, verbose=verbose)

    # 1. Reporte en consola (clase 1 = Renuncia voluntaria)
    analyzer.reporte_consola(X_test, clase_idx=1, top_n=15)

    # 2. Graficos para las 3 clases (summary + bar)
    print("\n  Generando graficos SHAP...")
    print("  " + "-" * 70)

    for clase_idx in range(3):
        r1 = analyzer.plot_summary(X_test, clase_idx=clase_idx, max_display=15)
        print(f"    {r1.name}")
        # CORREGIDO: plot_importancia_bar NO recibe X
        r2 = analyzer.plot_importancia_bar(clase_idx=clase_idx, top_n=15)
        print(f"    {r2.name}")

    # 3. Dependence plots para las 5 features mas importantes (clase 1)
    top5 = analyzer.importancia_global(clase_idx=1).head(5)["Feature"].tolist()
    for feature in top5:
        try:
            # CORREGIDO: plot_dependence SI recibe X
            r = analyzer.plot_dependence(feature, X_test, clase_idx=1)
            print(f"    {r.name}")
        except Exception as e:
            print(f"    [SKIP] {feature}: {e}")

    # 4. Waterfall para el registro de mayor riesgo de renuncia voluntaria
    proba_voluntaria = modelo.predict_proba(X_test)[:, 1]
    idx_max = int(np.argmax(proba_voluntaria))
    r_wf = analyzer.plot_waterfall_individual(X_test, idx_max, clase_idx=1)
    print(f"    {r_wf.name}")

    print(f"\n  Todos los graficos guardados en: {FIGURES_DIR}")

    return analyzer, idx_max, proba_voluntaria


if __name__ == "__main__":
    from data_pipeline.data_loader import cargar_datos
    from data_pipeline.preprocessing import preparar_datos
    import joblib
    from config import MODELS_DIR

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preparar_datos(df, verbose=False)

    modelo = joblib.load(MODELS_DIR / "modelo_final.pkl")
    features = preprocessor.get_feature_names()

    ejecutar_shap(modelo, X_train, X_test, features)