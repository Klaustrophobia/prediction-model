import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (
    TARGET, TARGET_MAP, TARGET_INV_MAP,
    ALERT_THRESHOLD, MEDIUM_THRESHOLD,
    REPORTS_DIR,
)


RECOMENDACIONES = {
    "Ratio_Promocion_Anios": "Revisar plan de carrera y considerar promocion o cambio de responsabilidades",
    "Ratio_Salario_Antiguedad": "Evaluar ajuste salarial acorde a la antiguedad y mercado",
    "Antiguedad_Meses": "Disenar plan de retencion para empleados con esta antiguedad (riesgo de estancamiento)",
    "Tiempo_Desplazamiento_Min": "Evaluar opcion de teletrabajo parcial o reubicacion",
    "Distancia_Domicilio_Trabajo_km": "Evaluar opcion de teletrabajo parcial o apoyo de transporte",
    "Ratio_Distancia_Tiempo": "Analizar alternativas de flexibilidad geografica",
    "Programa_Desarrollo": "Revisar si el empleado esta sobre-cualificado o requiere nuevos desafios",
    "Score_Insatisfaccion": "Agendar reunion 1:1 para identificar focos de insatisfaccion",
    "Score_Burnout": "Revisar carga laboral, redistribuir proyectos y considerar dias de descanso",
    "Carga_Trabajo_1a5": "Redistribuir carga laboral entre el equipo",
    "Carga_Neta": "Reequilibrar asignaciones y reducir horas extra",
    "Ratio_Capacitacion_Antiguedad": "Ofrecer capacitacion especializada o certificaciones",
    "Ratio_Ausencias_Antiguedad": "Investigar causas de ausentismo (clima, salud, carga)",
    "Evaluacion_Desempeno_1a5": "Disenar plan de mejora con acompanamiento",
    "Horas_Extra_Mes": "Revisar asignacion de horas extra y compensacion",
    "Dias_Vacaciones_Pendientes": "Facilitar toma de vacaciones pendientes",
    "Promociones": "Evaluar reconocimiento o nuevo rol",
    "Salario_Mensual_HNL": "Benchmark salarial con mercado y evaluar ajuste",
    "Universidad_Procedencia": "Analizar si hay brecha con expectativas del perfil",
    "Cargo": "Revisar encaje del rol actual con expectativas del empleado",
    "Historial_Disciplinario": "Acompanamiento con liderazgo y plan de mejora",
}

RECOMENDACION_GENERICA = "Analizar en reunion 1:1 con el empleado"


class AlertSystem:
    """
    Genera alertas de rotacion para todos los empleados.
    """

    def __init__(
        self,
        modelo,
        preprocessor,
        shap_analyzer,
        alert_threshold: float = ALERT_THRESHOLD,
        medium_threshold: float = MEDIUM_THRESHOLD,
    ):
        """
        Args:
            modelo: modelo entrenado
            preprocessor: DataPreprocessor ajustado (con feature_engineer + ColumnTransformer)
            shap_analyzer: instancia de ShapAnalyzer (ya con modelo cargado)
            alert_threshold: umbral para riesgo ALTO
            medium_threshold: umbral para riesgo MODERADO
        """
        self.modelo = modelo
        self.preprocessor = preprocessor
        self.shap_analyzer = shap_analyzer
        self.alert_threshold = alert_threshold
        self.medium_threshold = medium_threshold
        self.resultados_ = None
        self.reporte_ = None

    # -----------------------------------------------------
    # 1. Predecir riesgo para todos los empleados
    # -----------------------------------------------------
    def predecir_riesgo(self, df_empleados: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Calcula probabilidad de renuncia voluntaria para todos los empleados.
        Retorna un DataFrame con: id, probabilidades, nivel de riesgo.
        """
        if verbose:
            print(f"\n  Calculando riesgo para {len(df_empleados)} empleados...")

        # Guardar identificadores (fuera del preprocesamiento)
        ids = df_empleados["ID_Empleado"].values

        # Preprocesar todas las filas
        X_proc = self.preprocessor.transform(df_empleados)

        # Predicciones
        probas = self.modelo.predict_proba(X_proc)

        # Guardamos probabilidades por clase
        df_out = pd.DataFrame({
            "ID_Empleado": ids,
            "Prob_Permanencia": probas[:, 0],
            "Prob_Renuncia_Voluntaria": probas[:, 1],
            "Prob_Renuncia_Involuntaria": probas[:, 2],
        })

        # Nivel de riesgo (foco en renuncia voluntaria)
        p = df_out["Prob_Renuncia_Voluntaria"]
        df_out["Nivel_Riesgo"] = np.select(
            [p >= self.alert_threshold,
             p >= self.medium_threshold],
            ["Alto", "Moderado"],
            default="Bajo",
        )

        # Agregar columnas utiles del original para el reporte
        cols_contexto = [
            "ID_Empleado", "Firma_Auditora", "Cargo", "Area_Funcional",
            "Nivel_Jerarquico", "Antiguedad_Meses", "Salario_Mensual_HNL",
            "Edad", "Genero", "Evaluacion_Desempeno_1a5",
            "Rotacion_Personal",
        ]
        cols_disponibles = [c for c in cols_contexto if c in df_empleados.columns]
        df_out = df_out.merge(df_empleados[cols_disponibles], on="ID_Empleado", how="left")

        # X procesado se guarda como atributo para SHAP individual
        self.X_procesado_ = X_proc
        self.df_original_ = df_empleados.reset_index(drop=True)
        self.resultados_ = df_out

        if verbose:
            conteo = df_out["Nivel_Riesgo"].value_counts()
            print(f"    Riesgo ALTO:       {conteo.get('Alto', 0)}")
            print(f"    Riesgo MODERADO:   {conteo.get('Moderado', 0)}")
            print(f"    Riesgo BAJO:       {conteo.get('Bajo', 0)}")

        return df_out

    # -----------------------------------------------------
    # 2. Factores y recomendaciones para un empleado
    # -----------------------------------------------------
    def explicar_empleado(self, idx_fila: int, n: int = 5) -> dict:
        """
        Devuelve los factores que empujan hacia renuncia voluntaria
        y las recomendaciones asociadas para una fila del DataFrame original.
        """
        factores = self.shap_analyzer.top_factores_individual(
            self.X_procesado_, idx_fila, clase_idx=1, n=n
        )

        recomendaciones = []
        for f in factores["empuja_hacia"]:
            rec = RECOMENDACIONES.get(f["feature"], RECOMENDACION_GENERICA)
            recomendaciones.append({
                "feature": f["feature"],
                "shap": f["shap"],
                "recomendacion": rec,
            })

        return {
            "factores_riesgo": factores["empuja_hacia"],
            "factores_proteccion": factores["empuja_contra"],
            "recomendaciones": recomendaciones,
        }

    # -----------------------------------------------------
    # 3. Generar reporte completo
    # -----------------------------------------------------
    def generar_reporte(self, top_n: int = 20, verbose: bool = True) -> dict:
        """
        Construye un reporte con:
          - Resumen ejecutivo
          - Top N empleados en riesgo ALTO con factores y recomendaciones
        """
        if self.resultados_ is None:
            raise ValueError("Primero llama a predecir_riesgo().")

        df = self.resultados_
        df_alto = df[df["Nivel_Riesgo"] == "Alto"].sort_values(
            "Prob_Renuncia_Voluntaria", ascending=False
        )

        resumen = {
            "total_empleados": len(df),
            "riesgo_alto": int((df["Nivel_Riesgo"] == "Alto").sum()),
            "riesgo_moderado": int((df["Nivel_Riesgo"] == "Moderado").sum()),
            "riesgo_bajo": int((df["Nivel_Riesgo"] == "Bajo").sum()),
            "porcentaje_alto": round(
                (df["Nivel_Riesgo"] == "Alto").mean() * 100, 2
            ),
            "prob_promedio_voluntaria": round(
                df["Prob_Renuncia_Voluntaria"].mean(), 4
            ),
            "riesgo_por_firma": df.groupby("Firma_Auditora")["Nivel_Riesgo"]
                                  .apply(lambda s: (s == "Alto").sum())
                                  .to_dict() if "Firma_Auditora" in df.columns else {},
            "riesgo_por_area": df.groupby("Area_Funcional")["Nivel_Riesgo"]
                                 .apply(lambda s: (s == "Alto").sum())
                                 .to_dict() if "Area_Funcional" in df.columns else {},
            "riesgo_por_nivel": df.groupby("Nivel_Jerarquico")["Nivel_Riesgo"]
                                  .apply(lambda s: (s == "Alto").sum())
                                  .to_dict() if "Nivel_Jerarquico" in df.columns else {},
        }

        # Detalle de los top N empleados de riesgo ALTO
        detalle = []
        for _, row in df_alto.head(top_n).iterrows():
            idx_fila = self.df_original_[
                self.df_original_["ID_Empleado"] == row["ID_Empleado"]
            ].index[0]

            explicacion = self.explicar_empleado(idx_fila, n=5)

            detalle.append({
                "id": row["ID_Empleado"],
                "firma": row.get("Firma_Auditora", "-"),
                "cargo": row.get("Cargo", "-"),
                "area": row.get("Area_Funcional", "-"),
                "nivel": row.get("Nivel_Jerarquico", "-"),
                "antiguedad": row.get("Antiguedad_Meses", "-"),
                "salario": row.get("Salario_Mensual_HNL", "-"),
                "prob_voluntaria": row["Prob_Renuncia_Voluntaria"],
                "nivel_riesgo": row["Nivel_Riesgo"],
                "factores_riesgo": explicacion["factores_riesgo"],
                "factores_proteccion": explicacion["factores_proteccion"],
                "recomendaciones": explicacion["recomendaciones"],
            })

        self.reporte_ = {"resumen": resumen, "detalle": detalle}

        if verbose:
            self._imprimir_reporte_consola()

        return self.reporte_

    # -----------------------------------------------------
    # 4. Impresion en consola
    # -----------------------------------------------------
    def _imprimir_reporte_consola(self):
        if self.reporte_ is None:
            return

        r = self.reporte_["resumen"]

        print("\n" + "=" * 78)
        print("PASO 7: SISTEMA DE ALERTAS DE ROTACION")
        print("=" * 78)

        print("\n  RESUMEN EJECUTIVO")
        print("  " + "-" * 70)
        print(f"    Total de empleados analizados:    {r['total_empleados']}")
        print(f"    Riesgo ALTO:                      {r['riesgo_alto']} "
              f"({r['porcentaje_alto']}%)")
        print(f"    Riesgo MODERADO:                  {r['riesgo_moderado']}")
        print(f"    Riesgo BAJO:                      {r['riesgo_bajo']}")
        print(f"    Probabilidad promedio (voluntaria): {r['prob_promedio_voluntaria']}")

        if r["riesgo_por_firma"]:
            print("\n  Riesgo ALTO por firma:")
            for k, v in r["riesgo_por_firma"].items():
                print(f"    {k:<15} {v}")

        if r["riesgo_por_area"]:
            print("\n  Riesgo ALTO por area funcional:")
            for k, v in sorted(r["riesgo_por_area"].items(), key=lambda x: -x[1]):
                print(f"    {k:<25} {v}")

        if r["riesgo_por_nivel"]:
            print("\n  Riesgo ALTO por nivel jerarquico:")
            for k, v in sorted(r["riesgo_por_nivel"].items(), key=lambda x: -x[1]):
                print(f"    {k:<20} {v}")

        # Detalle top N
        print("\n" + "=" * 78)
        print(f"  TOP {len(self.reporte_['detalle'])} EMPLEADOS EN RIESGO ALTO DE "
              f"RENUNCIA VOLUNTARIA")
        print("=" * 78)

        for i, emp in enumerate(self.reporte_["detalle"], 1):
            print(f"\n  [{i}] {emp['id']} | {emp['firma']} | {emp['cargo']}")
            print(f"      Area: {emp['area']} | Nivel: {emp['nivel']} | "
                  f"Antiguedad: {emp['antiguedad']} meses")
            print(f"      Probabilidad de renuncia voluntaria: "
                  f"{emp['prob_voluntaria']:.2%}")

            print(f"\n      Factores que EMPUJAN hacia la renuncia:")
            for f in emp["factores_riesgo"]:
                print(f"        + {f['feature']:<38} SHAP = {f['shap']:+.4f}")

            print(f"\n      Factores que PROTEGEN (empujan a permanecer):")
            if emp["factores_proteccion"]:
                for f in emp["factores_proteccion"]:
                    print(f"        - {f['feature']:<38} SHAP = {f['shap']:+.4f}")
            else:
                print("        (ninguno)")

            print(f"\n      Recomendaciones de RRHH:")
            for rec in emp["recomendaciones"]:
                print(f"        - {rec['recomendacion']}")

            print("  " + "-" * 74)

    # -----------------------------------------------------
    # 5. Guardar reporte en archivo .txt
    # -----------------------------------------------------
    def guardar_reporte(self, ruta: Path = None) -> Path:
        """
        Guarda el reporte completo en un .txt legible.
        """
        if self.reporte_ is None:
            raise ValueError("Primero llama a generar_reporte().")

        if ruta is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta = REPORTS_DIR / f"alertas_{timestamp}.txt"

        r = self.reporte_["resumen"]
        lineas = []

        lineas.append("=" * 78)
        lineas.append("REPORTE DE ALERTAS DE ROTACION")
        lineas.append("=" * 78)
        lineas.append(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lineas.append("")

        lineas.append("-" * 78)
        lineas.append("RESUMEN EJECUTIVO")
        lineas.append("-" * 78)
        lineas.append(f"Total de empleados analizados:      {r['total_empleados']}")
        lineas.append(f"Riesgo ALTO:                        {r['riesgo_alto']} ({r['porcentaje_alto']}%)")
        lineas.append(f"Riesgo MODERADO:                    {r['riesgo_moderado']}")
        lineas.append(f"Riesgo BAJO:                        {r['riesgo_bajo']}")
        lineas.append(f"Probabilidad promedio (voluntaria): {r['prob_promedio_voluntaria']}")
        lineas.append("")

        if r["riesgo_por_firma"]:
            lineas.append("Riesgo ALTO por firma:")
            for k, v in r["riesgo_por_firma"].items():
                lineas.append(f"  {k:<15} {v}")
            lineas.append("")

        if r["riesgo_por_area"]:
            lineas.append("Riesgo ALTO por area funcional:")
            for k, v in sorted(r["riesgo_por_area"].items(), key=lambda x: -x[1]):
                lineas.append(f"  {k:<25} {v}")
            lineas.append("")

        if r["riesgo_por_nivel"]:
            lineas.append("Riesgo ALTO por nivel jerarquico:")
            for k, v in sorted(r["riesgo_por_nivel"].items(), key=lambda x: -x[1]):
                lineas.append(f"  {k:<20} {v}")
            lineas.append("")

        lineas.append("=" * 78)
        lineas.append(f"TOP {len(self.reporte_['detalle'])} EMPLEADOS EN RIESGO ALTO")
        lineas.append("=" * 78)

        for i, emp in enumerate(self.reporte_["detalle"], 1):
            lineas.append("")
            lineas.append(f"[{i}] {emp['id']} | {emp['firma']} | {emp['cargo']}")
            lineas.append(f"    Area: {emp['area']} | Nivel: {emp['nivel']} | "
                          f"Antiguedad: {emp['antiguedad']} meses")
            lineas.append(f"    Probabilidad de renuncia voluntaria: "
                          f"{emp['prob_voluntaria']:.2%}")
            lineas.append("")
            lineas.append("    Factores que EMPUJAN hacia la renuncia:")
            for f in emp["factores_riesgo"]:
                lineas.append(f"      + {f['feature']:<38} SHAP = {f['shap']:+.4f}")
            lineas.append("")
            lineas.append("    Factores que PROTEGEN:")
            if emp["factores_proteccion"]:
                for f in emp["factores_proteccion"]:
                    lineas.append(f"      - {f['feature']:<38} SHAP = {f['shap']:+.4f}")
            else:
                lineas.append("      (ninguno)")
            lineas.append("")
            lineas.append("    Recomendaciones de RRHH:")
            for rec in emp["recomendaciones"]:
                lineas.append(f"      - {rec['recomendacion']}")
            lineas.append("-" * 74)

        lineas.append("")
        lineas.append("=" * 78)
        lineas.append("FIN DEL REPORTE")
        lineas.append("=" * 78)

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))

        print(f"\n  Reporte guardado en: {ruta}")
        return ruta

    # -----------------------------------------------------
    # 6. Exportar a CSV (opcional, para RRHH)
    # -----------------------------------------------------
    def exportar_csv(self, ruta: Path = None) -> Path:
        """
        Exporta la tabla de riesgos a CSV para uso en Excel/Power BI.
        """
        if self.resultados_ is None:
            raise ValueError("Primero llama a predecir_riesgo().")

        if ruta is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta = REPORTS_DIR / f"riesgos_empleados_{timestamp}.csv"

        self.resultados_.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"  CSV exportado en: {ruta}")
        return ruta


# ============================================================
# FUNCION DE ALTO NIVEL PARA USAR EN main.py
# ============================================================
def ejecutar_alertas(
    modelo,
    preprocessor,
    shap_analyzer,
    df_empleados: pd.DataFrame,
    top_n: int = 20,
    verbose: bool = True,
):
    """
    Ejecuta el PASO 7 completo:
      1. Predice riesgo para todos
      2. Genera reporte con top N de alto riesgo
      3. Imprime en consola
      4. Guarda .txt y .csv en artifacts/reports/
    Retorna el AlertSystem configurado.
    """
    sistema = AlertSystem(modelo, preprocessor, shap_analyzer)

    sistema.predecir_riesgo(df_empleados, verbose=verbose)
    sistema.generar_reporte(top_n=top_n, verbose=verbose)
    sistema.guardar_reporte()
    sistema.exportar_csv()

    return sistema


if __name__ == "__main__":
    # Ejecucion directa para pruebas
    import joblib
    from config import MODELS_DIR
    from data_pipeline.data_loader import cargar_datos
    from explainability.shap_analyzer import ShapAnalyzer

    df = cargar_datos()
    modelo = joblib.load(MODELS_DIR / "modelo_final.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    features = preprocessor.get_feature_names()

    analyzer = ShapAnalyzer(modelo, features)
    X_proc = preprocessor.transform(df)
    analyzer.calcular(X_proc, verbose=False)

    sistema = ejecutar_alertas(modelo, preprocessor, analyzer, df, top_n=10)