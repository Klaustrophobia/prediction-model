import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI (evita problemas en Windows headless)
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from config import FIGURES_DIR, TARGET_INV_MAP


class ShapAnalyzer:
   

    def __init__(self, modelo, feature_names, class_names=None):
       
        self.modelo = modelo
        self.feature_names = list(feature_names)
        self.class_names = class_names or [TARGET_INV_MAP[i] for i in range(3)]
        self.explainer = None
        self.shap_values = None

   
    def calcular(self, X, verbose: bool = True):
       
        if verbose:
            print(f"\n  Calculando SHAP values sobre {X.shape[0]} registros...")

        self.explainer = shap.TreeExplainer(self.modelo)
        shap_values = self.explainer.shap_values(X)

       
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = [np.asarray(sv) for sv in shap_values]

        self.shap_values = shap_values

        if verbose:
            n_clases = len(shap_values)
            n_features = shap_values[0].shape[1]
            print(f"  SHAP calculado: {n_clases} clases x {n_features} features")

        return shap_values

   
    def importancia_global(self, clase_idx: int = 1) -> pd.DataFrame:
        
        if self.shap_values is None:
            raise ValueError("Primero llama a calcular().")

        valores = np.abs(self.shap_values[clase_idx]).mean(axis=0)
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importancia_SHAP": valores,
        }).sort_values("Importancia_SHAP", ascending=False).reset_index(drop=True)

        return df

   
    def plot_summary(self, X, clase_idx: int = 1, max_display: int = 20) -> Path:
   
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

    
    def plot_importancia_bar(self, clase_idx: int = 1, top_n: int = 20) -> Path:
       
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

   
    def plot_dependence(self, feature_name: str, X, clase_idx: int = 1) -> Path:
        
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


    def plot_waterfall_individual(self, X, idx_registro: int, clase_idx: int = 1,
                                  max_display: int = 12) -> Path:
       
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

   
    def top_factores_individual(self, X, idx_registro: int, clase_idx: int = 1,
                                 n: int = 5) -> dict:
    
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


    def reporte_consola(self, X, clase_idx: int = 1, top_n: int = 15):
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



def ejecutar_shap(modelo, X_train, X_test, feature_names, verbose: bool = True):
    print("\n" + "=" * 78)
    print("PASO 6: EXPLICABILIDAD CON SHAP")
    print("=" * 78)

    analyzer = ShapAnalyzer(modelo, feature_names)
    analyzer.calcular(X_test, verbose=verbose)

    analyzer.reporte_consola(X_test, clase_idx=1, top_n=15)

    print("\n  Generando graficos SHAP...")
    print("  " + "-" * 70)

    for clase_idx in range(3):
        r1 = analyzer.plot_summary(X_test, clase_idx=clase_idx, max_display=15)
        print(f"    {r1.name}")
        r2 = analyzer.plot_importancia_bar(clase_idx=clase_idx, top_n=15)
        print(f"    {r2.name}")

    top5 = analyzer.importancia_global(clase_idx=1).head(5)["Feature"].tolist()
    for feature in top5:
        try:
            # CORREGIDO: plot_dependence SI recibe X
            r = analyzer.plot_dependence(feature, X_test, clase_idx=1)
            print(f"    {r.name}")
        except Exception as e:
            print(f"    [SKIP] {feature}: {e}")

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