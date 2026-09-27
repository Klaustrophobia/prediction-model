
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

from config import (
    MODELS_DIR, REPORTS_DIR, TARGET_INV_MAP,
    ALERT_THRESHOLD, MEDIUM_THRESHOLD,
)
from data_pipeline.data_loader import cargar_datos
from alerts.alert_system import RECOMENDACIONES, RECOMENDACION_GENERICA


st.set_page_config(
    page_title="Dashboard Rotacion - Firmas Auditoras",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-alto {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .metric-card-medio {
        background: linear-gradient(135deg, #f39c12 0%, #d68910 100%);
    }
    .metric-card-bajo {
        background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-alto {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-moderado {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-bajo {
        color: #27ae60;
        font-weight: bold;
    }
    .factor-card {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #3498db;
    }
    .factor-riesgo {
        border-left-color: #e74c3c;
    }
    .factor-proteccion {
        border-left-color: #27ae60;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo_y_preprocessor():
    modelo = joblib.load(MODELS_DIR / "modelo_final.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return modelo, preprocessor


@st.cache_data
def cargar_datos_cache():
    return cargar_datos(verbose=False)


@st.cache_data
def calcular_predicciones_todos(_modelo, _preprocessor, df):
    X_proc = _preprocessor.transform(df)
    probas = _modelo.predict_proba(X_proc)

    df_out = df.copy()
    df_out["Prob_Permanencia"] = probas[:, 0]
    df_out["Prob_Renuncia_Voluntaria"] = probas[:, 1]
    df_out["Prob_Renuncia_Involuntaria"] = probas[:, 2]

    p = df_out["Prob_Renuncia_Voluntaria"]
    df_out["Nivel_Riesgo"] = np.select(
        [p >= ALERT_THRESHOLD, p >= MEDIUM_THRESHOLD],
        ["Alto", "Moderado"],
        default="Bajo",
    )
    df_out["Prediccion"] = np.argmax(probas, axis=1)
    df_out["Prediccion_Clase"] = df_out["Prediccion"].map(TARGET_INV_MAP)
    return df_out


def render_metric_card(label, value, tipo="default"):
    clase_extra = ""
    if tipo == "alto":
        clase_extra = "metric-card-alto"
    elif tipo == "medio":
        clase_extra = "metric-card-medio"
    elif tipo == "bajo":
        clase_extra = "metric-card-bajo"

    st.markdown(f"""
    <div class="metric-card {clase_extra}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def color_riesgo(nivel):
    if nivel == "Alto":
        return "risk-alto"
    elif nivel == "Moderado":
        return "risk-moderado"
    else:
        return "risk-bajo"


def predecir_individual(modelo, preprocessor, datos: dict):
    import shap

    df_input = pd.DataFrame([datos])

    from config import NUM_FEATURES, CAT_FEATURES
    for col in NUM_FEATURES + CAT_FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0 if col in NUM_FEATURES else "Desconocido"

    X_proc = preprocessor.transform(df_input)
    proba = modelo.predict_proba(X_proc)[0]

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    valores_vol = shap_values[1][0]

    feature_names = preprocessor.get_feature_names()
    pares = list(zip(feature_names, valores_vol, X_proc[0]))
    ordenados = sorted(pares, key=lambda t: abs(t[1]), reverse=True)[:5]

    factores_riesgo = []
    factores_proteccion = []
    recomendaciones = []

    for feat, val_shap, val_feat in ordenados:
        item = {
            "feature": feat,
            "shap": float(val_shap),
            "valor": float(val_feat) if not isinstance(val_feat, str) else 0.0,
        }
        if val_shap > 0:
            factores_riesgo.append(item)
            recomendaciones.append(RECOMENDACIONES.get(feat, RECOMENDACION_GENERICA))
        else:
            factores_proteccion.append(item)

    return {
        "proba": proba,
        "factores_riesgo": factores_riesgo,
        "factores_proteccion": factores_proteccion,
        "recomendaciones": recomendaciones,
    }


def render_sidebar(df):
    st.sidebar.markdown("## Panel de Control")
    st.sidebar.markdown("---")

    firmas = ["Todas"] + sorted(df["Firma_Auditora"].unique().tolist())
    firma_sel = st.sidebar.selectbox("Firma auditora", firmas, index=0)

    areas = ["Todas"] + sorted(df["Area_Funcional"].unique().tolist())
    area_sel = st.sidebar.selectbox("Area funcional", areas, index=0)

    niveles = ["Todos"] + sorted(df["Nivel_Jerarquico"].unique().tolist())
    nivel_sel = st.sidebar.selectbox("Nivel jerarquico", niveles, index=0)

    riesgos = ["Todos", "Alto", "Moderado", "Bajo"]
    riesgo_sel = st.sidebar.selectbox("Nivel de riesgo", riesgos, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Umbrales del modelo")
    st.sidebar.markdown(f"- Riesgo **Alto**: p >= {ALERT_THRESHOLD:.2f}")
    st.sidebar.markdown(f"- Riesgo **Moderado**: p >= {MEDIUM_THRESHOLD:.2f}")
    st.sidebar.markdown(f"- Riesgo **Bajo**: p < {MEDIUM_THRESHOLD:.2f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Modelo: XGBoost optimizado con Optuna*")

    return firma_sel, area_sel, nivel_sel, riesgo_sel


def aplicar_filtros(df, firma, area, nivel, riesgo):
    df_f = df.copy()
    if firma != "Todas":
        df_f = df_f[df_f["Firma_Auditora"] == firma]
    if area != "Todas":
        df_f = df_f[df_f["Area_Funcional"] == area]
    if nivel != "Todos":
        df_f = df_f[df_f["Nivel_Jerarquico"] == nivel]
    if riesgo != "Todos":
        df_f = df_f[df_f["Nivel_Riesgo"] == riesgo]
    return df_f



def seccion_resumen(df, df_filtrado):
    st.markdown("## Resumen Ejecutivo")

    total = len(df_filtrado)
    alto = (df_filtrado["Nivel_Riesgo"] == "Alto").sum()
    moderado = (df_filtrado["Nivel_Riesgo"] == "Moderado").sum()
    bajo = (df_filtrado["Nivel_Riesgo"] == "Bajo").sum()
    prob_prom = df_filtrado["Prob_Renuncia_Voluntaria"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_metric_card("Total empleados", total, "default")
    with col2:
        pct = (alto / total * 100) if total > 0 else 0
        render_metric_card("Riesgo alto", f"{alto} ({pct:.1f}%)", "alto")
    with col3:
        pct = (moderado / total * 100) if total > 0 else 0
        render_metric_card("Riesgo moderado", f"{moderado} ({pct:.1f}%)", "medio")
    with col4:
        pct = (bajo / total * 100) if total > 0 else 0
        render_metric_card("Riesgo bajo", f"{bajo} ({pct:.1f}%)", "bajo")
    with col5:
        render_metric_card("Prob. prom. renuncia", f"{prob_prom:.1%}", "default")

    st.markdown("---")


def seccion_distribucion(df_filtrado):
    st.markdown("## Distribucion del Riesgo")

    col1, col2 = st.columns(2)

    with col1:
        conteo = df_filtrado["Nivel_Riesgo"].value_counts().reindex(
            ["Alto", "Moderado", "Bajo"], fill_value=0
        ).reset_index()
        conteo.columns = ["Nivel_Riesgo", "Cantidad"]

        colores = {"Alto": "#e74c3c", "Moderado": "#f39c12", "Bajo": "#27ae60"}
        fig = px.pie(
            conteo, values="Cantidad", names="Nivel_Riesgo",
            color="Nivel_Riesgo", color_discrete_map=colores,
            title="Distribucion por Nivel de Riesgo",
            hole=0.4,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_alto = df_filtrado[df_filtrado["Nivel_Riesgo"] == "Alto"]
        if len(df_alto) > 0:
            conteo = df_alto.groupby("Firma_Auditora").size().reset_index(name="Alto riesgo")
            fig = px.bar(
                conteo, x="Firma_Auditora", y="Alto riesgo",
                title="Empleados en Riesgo Alto por Firma",
                color="Firma_Auditora",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay empleados en riesgo alto con los filtros actuales.")

    col1, col2 = st.columns(2)

    with col1:
        conteo = df_filtrado.groupby("Area_Funcional")["Nivel_Riesgo"].apply(
            lambda s: (s == "Alto").sum()
        ).reset_index(name="Alto riesgo").sort_values("Alto riesgo", ascending=True)

        fig = px.bar(
            conteo, x="Alto riesgo", y="Area_Funcional", orientation="h",
            title="Empleados en Riesgo Alto por Area Funcional",
            color="Alto riesgo", color_continuous_scale="Reds",
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        conteo = df_filtrado.groupby("Nivel_Jerarquico")["Nivel_Riesgo"].apply(
            lambda s: (s == "Alto").sum()
        ).reset_index(name="Alto riesgo").sort_values("Alto riesgo", ascending=True)

        fig = px.bar(
            conteo, x="Alto riesgo", y="Nivel_Jerarquico", orientation="h",
            title="Empleados en Riesgo Alto por Nivel Jerarquico",
            color="Alto riesgo", color_continuous_scale="Oranges",
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


def seccion_tabla_riesgo(df_filtrado, top_n=20):
    st.markdown("## Top Empleados en Riesgo de Renuncia Voluntaria")

    df_alto = df_filtrado[df_filtrado["Nivel_Riesgo"] == "Alto"].sort_values(
        "Prob_Renuncia_Voluntaria", ascending=False
    ).head(top_n)

    if len(df_alto) == 0:
        st.info("No hay empleados en riesgo alto con los filtros actuales.")
        return

    tabla = df_alto[[
        "ID_Empleado", "Firma_Auditora", "Cargo", "Area_Funcional",
        "Nivel_Jerarquico", "Antiguedad_Meses", "Salario_Mensual_HNL",
        "Prob_Renuncia_Voluntaria", "Nivel_Riesgo",
    ]].copy()

    tabla["Prob_Renuncia_Voluntaria"] = (tabla["Prob_Renuncia_Voluntaria"] * 100).round(2)
    tabla = tabla.rename(columns={
        "Prob_Renuncia_Voluntaria": "Prob. Renuncia (%)",
        "Salario_Mensual_HNL": "Salario (HNL)",
        "Antiguedad_Meses": "Antiguedad (meses)",
    })

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Prob. Renuncia (%)": st.column_config.ProgressColumn(
                "Prob. Renuncia (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    csv = df_alto.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Descargar lista completa (CSV)",
        data=csv,
        file_name=f"empleados_riesgo_alto_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.markdown("---")


def seccion_prediccion_individual(modelo, preprocessor, df):
    st.markdown("## Prediccion Individual en Tiempo Real")

    with st.expander("Seleccionar empleado de la base o ingresar manualmente", expanded=False):
        opcion = st.radio(
            "Modo de entrada",
            ["Seleccionar de la base", "Ingreso manual"],
            horizontal=True,
        )

        if opcion == "Seleccionar de la base":
            ids = df["ID_Empleado"].tolist()
            id_sel = st.selectbox("Selecciona un empleado", ids)
            empleado = df[df["ID_Empleado"] == id_sel].iloc[0].to_dict()

            st.write("**Datos del empleado seleccionado:**")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Edad", int(empleado["Edad"]))
            with cols[1]:
                st.metric("Antiguedad", f"{empleado['Antiguedad_Meses']:.0f} meses")
            with cols[2]:
                st.metric("Salario", f"L {empleado['Salario_Mensual_HNL']:,.0f}")
            with cols[3]:
                st.metric("Cargo", str(empleado["Cargo"])[:20])

        else:
            with st.form("form_prediccion"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    firma = st.selectbox("Firma", ["Deloitte", "KPMG", "PwC"])
                    genero = st.selectbox("Genero", ["Masculino", "Femenino"])
                    estado_civil = st.selectbox("Estado civil", ["Soltero/a", "Casado/a", "Union libre", "Divorciado/a"])
                    nivel_edu = st.selectbox("Nivel educativo", ["Licenciatura", "Maestria", "Doctorado"])
                    universidad = st.selectbox("Universidad", ["UNITEC", "UNAH", "UNICAH", "UJCV", "UMH", "UPNFM"])
                    cargo = st.text_input("Cargo", value="Asistente de Auditoria")
                    area = st.selectbox("Area funcional", ["Auditoria", "Impuestos", "Consultoria", "Apoyo administrativo"])
                    nivel_jer = st.selectbox("Nivel jerarquico", ["Operativo", "Profesional", "Mando medio", "Gerencial", "Directivo"])
                    programa = st.selectbox("Programa desarrollo", ["Si", "No"])
                    historial = st.selectbox("Historial disciplinario", ["Ninguno", "Llamado de atencion", "Amonestacion escrita"])

                with col2:
                    edad = st.number_input("Edad", 18, 70, 30)
                    antiguedad = st.number_input("Antiguedad (meses)", 0.0, 240.0, 24.0)
                    salario = st.number_input("Salario (HNL)", 0.0, 200000.0, 25000.0)
                    incremento = st.number_input("Incremento (%)", 0.0, 50.0, 5.0)
                    evaluacion = st.slider("Evaluacion (1-5)", 1.0, 5.0, 3.5, 0.1)
                    horas_extra = st.number_input("Horas extra / mes", 0.0, 100.0, 5.0)
                    horas_cap = st.number_input("Horas capacitacion / anio", 0.0, 200.0, 40.0)

                with col3:
                    promociones = st.number_input("Promociones", 0, 10, 1)
                    ausencias = st.number_input("Ausencias / anio", 0, 50, 5)
                    vacaciones = st.number_input("Vacaciones pendientes (dias)", 0, 60, 5)
                    distancia = st.number_input("Distancia (km)", 0.0, 100.0, 10.0)
                    tiempo_despl = st.number_input("Tiempo desplazamiento (min)", 0.0, 180.0, 30.0)
                    carga = st.slider("Carga trabajo (1-5)", 1.0, 5.0, 3.0, 0.1)

                submitted = st.form_submit_button("Predecir")

            if submitted:
                empleado = {
                    "Firma_Auditora": firma,
                    "Genero": genero,
                    "Estado_Civil": estado_civil,
                    "Nivel_Educativo": nivel_edu,
                    "Universidad_Procedencia": universidad,
                    "Cargo": cargo,
                    "Area_Funcional": area,
                    "Nivel_Jerarquico": nivel_jer,
                    "Programa_Desarrollo": programa,
                    "Historial_Disciplinario": historial,
                    "Edad": edad,
                    "Antiguedad_Meses": antiguedad,
                    "Salario_Mensual_HNL": salario,
                    "Incremento_Salarial_Porcentaje": incremento,
                    "Evaluacion_Desempeno_1a5": evaluacion,
                    "Horas_Extra_Mes": horas_extra,
                    "Horas_Capacitacion_Anio": horas_cap,
                    "Promociones": promociones,
                    "Ausencias_Anio": ausencias,
                    "Dias_Vacaciones_Pendientes": vacaciones,
                    "Distancia_Domicilio_Trabajo_km": distancia,
                    "Tiempo_Desplazamiento_Min": tiempo_despl,
                    "Carga_Trabajo_1a5": carga,
                }
                mostrar_prediccion(modelo, preprocessor, empleado)


def mostrar_prediccion(modelo, preprocessor, empleado):
    with st.spinner("Calculando prediccion..."):
        resultado = predecir_individual(modelo, preprocessor, empleado)

    proba = resultado["proba"]
    p_vol = proba[1]

    if p_vol >= ALERT_THRESHOLD:
        nivel = "Alto"
    elif p_vol >= MEDIUM_THRESHOLD:
        nivel = "Moderado"
    else:
        nivel = "Bajo"

    st.markdown("### Resultado de la Prediccion")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Permanencia", f"{proba[0]:.1%}")
    with col2:
        st.metric("Renuncia voluntaria", f"{proba[1]:.1%}", delta=None)
    with col3:
        st.metric("Renuncia involuntaria", f"{proba[2]:.1%}")
    with col4:
        st.metric("Nivel de riesgo", nivel)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_vol * 100,
        title={"text": "Probabilidad de Renuncia Voluntaria (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if p_vol > ALERT_THRESHOLD else "#f39c12" if p_vol > MEDIUM_THRESHOLD else "#27ae60"},
            "steps": [
                {"range": [0, MEDIUM_THRESHOLD * 100], "color": "#d5f4e6"},
                {"range": [MEDIUM_THRESHOLD * 100, ALERT_THRESHOLD * 100], "color": "#fef5e7"},
                {"range": [ALERT_THRESHOLD * 100, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Explicacion SHAP")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factores que empujan hacia la renuncia:**")
        if resultado["factores_riesgo"]:
            for f in resultado["factores_riesgo"]:
                st.markdown(f"""
                <div class="factor-card factor-riesgo">
                    <strong>{f['feature']}</strong><br>
                    SHAP: <code>{f['shap']:+.4f}</code>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Sin factores de riesgo significativos.")

    with col2:
        st.markdown("**Factores que protegen (empujan a permanecer):**")
        if resultado["factores_proteccion"]:
            for f in resultado["factores_proteccion"]:
                st.markdown(f"""
                <div class="factor-card factor-proteccion">
                    <strong>{f['feature']}</strong><br>
                    SHAP: <code>{f['shap']:+.4f}</code>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Sin factores de proteccion significativos.")

    st.markdown("### Recomendaciones de RRHH")
    if resultado["recomendaciones"]:
        for i, rec in enumerate(resultado["recomendaciones"], 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No se generaron recomendaciones automaticas.")

    st.markdown("---")


def seccion_importancia_features():
    st.markdown("## Importancia Global de Features")

    ruta = REPORTS_DIR / "figures" / "07_shap_importancia_clase_1.png"
    if ruta.exists():
        st.image(str(ruta), caption="Top features - Renuncia Voluntaria")
    else:
        st.info("Ejecuta el PASO 6 para generar los graficos SHAP.")

    st.markdown("---")


def seccion_descargas():
    st.markdown("## Reportes Generados")

    reportes = list(REPORTS_DIR.glob("*.txt")) + list(REPORTS_DIR.glob("*.csv"))

    if not reportes:
        st.info("Aun no se han generado reportes. Ejecuta main.py primero.")
        return

    for r in sorted(reportes, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        col1, col2 = st.columns([3, 1])
        with col1:
            tamaño_kb = r.stat().st_size / 1024
            fecha = datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"**{r.name}** - {tamaño_kb:.1f} KB - {fecha}")
        with col2:
            with open(r, "rb") as f:
                st.download_button(
                    "Descargar",
                    f,
                    file_name=r.name,
                    key=f"dl_{r.name}",
                )


def main():
    # Header
    st.markdown('<div class="main-header">Dashboard de Rotacion de Personal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Modelo predictivo para firmas auditoras - KPMG, PwC, Deloitte</div>', unsafe_allow_html=True)

    # Cargar recursos
    try:
        modelo, preprocessor = cargar_modelo_y_preprocessor()
        df_original = cargar_datos_cache()
        df = calcular_predicciones_todos(modelo, preprocessor, df_original)
    except Exception as e:
        st.error(f"Error cargando modelo o datos: {e}")
        st.info("Asegurate de haber ejecutado `python main.py` al menos una vez.")
        st.stop()

    # Sidebar
    firma, area, nivel, riesgo = render_sidebar(df)

    # Aplicar filtros
    df_filtrado = aplicar_filtros(df, firma, area, nivel, riesgo)

    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen",
        "Distribucion",
        "Empleados en Riesgo",
        "Prediccion Individual",
        "Reportes",
    ])

    with tab1:
        seccion_resumen(df, df_filtrado)

    with tab2:
        seccion_distribucion(df_filtrado)

    with tab3:
        seccion_tabla_riesgo(df_filtrado, top_n=30)

    with tab4:
        seccion_prediccion_individual(modelo, preprocessor, df)

    with tab5:
        seccion_importancia_features()
        seccion_descargas()

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
        f"Modelo XGBoost | Ultima actualizacion: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Datos sinteticos para fines academicos"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()