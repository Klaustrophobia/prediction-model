import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_pipeline.data_loader import cargar_datos, validar_datos, imprimir_resumen
from data_pipeline.eda import ejecutar_eda
from data_pipeline.preprocessing import preparar_datos
from modeling.model_selector import BaselineModelSelector
from modeling.train import ModelTrainer
from explainability.shap_analyzer import ejecutar_shap
from alerts.alert_system import ejecutar_alertas
from ethics.bias_detection import ejecutar_analisis_sesgos


def main():
    print("\n" + "=" * 65)
    print("MODELO PREDICTIVO DE ROTACION - FIRMAS AUDITORAS")
    print("=" * 65)

    print("\n" + "-" * 65)
    print("PASO 1: CARGA Y VALIDACION DE DATOS")
    print("-" * 65)
    df = cargar_datos()
    resumen = validar_datos(df)
    imprimir_resumen(resumen)
    print("\n[OK] PASO 1 completado\n")

    print("\n" + "-" * 65)
    print("PASO 2: ANALISIS EXPLORATORIO DE DATOS (EDA)")
    print("-" * 65)
    ejecutar_eda(df)
    print("\n[OK] PASO 2 completado\n")

    print("\n" + "-" * 65)
    print("PASO 3: PREPROCESAMIENTO Y FEATURE ENGINEERING")
    print("-" * 65)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preparar_datos(df)
    print()
    preprocessor.save()
    print("\n[OK] PASO 3 completado\n")

    print("\n" + "-" * 65)
    print("PASO 4: ENTRENAMIENTO DE MODELOS BASELINE")
    print("-" * 65)
    selector = BaselineModelSelector()
    selector.entrenar_todos(X_train, y_train, X_val, y_val)

    print("\n" + "=" * 78)
    print("TABLA COMPARATIVA DE MODELOS BASELINE")
    print("=" * 78)
    print(selector.tabla_comparativa().to_string(index=False))

    mejor_nombre, _ = selector.mejor_modelo()
    print(f"\n  Mejor modelo baseline: {mejor_nombre}")
    print("\n[OK] PASO 4 completado\n")

    print("\n" + "-" * 65)
    print("PASO 5: OPTIMIZACION DE HIPERPARAMETROS")
    print("-" * 65)

    if mejor_nombre in ["XGBoost", "LightGBM"]:
        nombre_optimizar = mejor_nombre
    else:
        nombre_optimizar = "XGBoost"
        print("\n  Se optimizara XGBoost como modelo principal.\n")

    trainer = ModelTrainer(nombre_modelo=nombre_optimizar, n_trials=30)
    trainer.entrenar(X_train, y_train, verbose=True)
    trainer.evaluar(X_test, y_test)
    trainer.guardar()
    print("\n[OK] PASO 5 completado\n")

    print("\n" + "-" * 65)
    print("PASO 6: EXPLICABILIDAD CON SHAP")
    print("-" * 65)

    feature_names = preprocessor.get_feature_names()
    analyzer, idx_max, proba = ejecutar_shap(
        trainer.modelo_, X_train, X_test, feature_names, verbose=True
    )

    print(f"\n  Empleado de mayor riesgo (registro #{idx_max}): "
          f"prob = {proba[idx_max]:.4f}")
    print("\n[OK] PASO 6 completado\n")

    print("\n" + "-" * 65)
    print("PASO 7: SISTEMA DE ALERTAS")
    print("-" * 65)

    X_proc_all = preprocessor.transform(df)
    analyzer.calcular(X_proc_all, verbose=False)

    ejecutar_alertas(
        modelo=trainer.modelo_,
        preprocessor=preprocessor,
        shap_analyzer=analyzer,
        df_empleados=df,
        top_n=20,
        verbose=True,
    )

    print("\n[OK] PASO 7 completado\n")

    print("\n" + "-" * 65)
    print("PASO 8: ETICA Y ANALISIS DE SESGOS")
    print("-" * 65)

    ejecutar_analisis_sesgos(
        modelo=trainer.modelo_,
        preprocessor=preprocessor,
        df_empleados=df,
        verbose=True,
    )

    print("\n[OK] PASO 8 completado\n")


if __name__ == "__main__":
    main()
