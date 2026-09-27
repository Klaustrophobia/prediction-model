"""
Análisis Exploratorio de Datos (EDA).
Ubicación: data_pipeline/eda.py

Compatible con Windows / Linux / macOS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import (
    TARGET, FIGURES_DIR, NUM_FEATURES, CAT_FEATURES,
    COLS_EXCLUIR, TARGET_MAP,
)

# Configuración global de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 9


# ============================================================
# 1. RESUMEN ESTADÍSTICO
# ============================================================
def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas de variables numéricas."""
    return df[NUM_FEATURES].describe().T.round(2)


def analisis_target(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de la variable objetivo."""
    conteo = df[TARGET].value_counts()
    pct = df[TARGET].value_counts(normalize=True) * 100
    tabla = pd.DataFrame({"Conteo": conteo, "Porcentaje": pct.round(2)})
    return tabla


def tasas_por_categoria(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tasa de rotación (en %) por categoría de una variable."""
    tabla = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
    tabla["n"] = df[col].value_counts()
    return tabla.round(2)


# ============================================================
# 2. GRÁFICOS
# ============================================================
def guardar_grafico_target(df: pd.DataFrame) -> Path:
    """Distribución de la variable objetivo (barras + pastel)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Colores consistentes
    colores = {"Permanencia": "#2ecc71",
               "Renuncia voluntaria": "#e74c3c",
               "Renuncia involuntaria": "#f39c12"}

    # Barras
    orden = ["Permanencia", "Renuncia voluntaria", "Renuncia involuntaria"]
    conteo = df[TARGET].value_counts().reindex(orden)
    sns.barplot(x=conteo.index, y=conteo.values, ax=axes[0],
                palette=[colores[c] for c in conteo.index])
    axes[0].set_title("Distribución de la Variable Objetivo", fontweight="bold")
    axes[0].set_ylabel("Cantidad de empleados")
    axes[0].set_xlabel("")
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Pastel
    axes[1].pie(conteo.values, labels=conteo.index, autopct="%1.1f%%",
                colors=[colores[c] for c in conteo.index],
                startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Proporción de Clases", fontweight="bold")

    plt.tight_layout()
    ruta = FIGURES_DIR / "01_distribucion_target.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_grafico_numericas(df: pd.DataFrame) -> Path:
    """Histogramas de variables numéricas segmentados por target."""
    n = len(NUM_FEATURES)
    cols = 4
    filas = (n + cols - 1) // cols

    fig, axes = plt.subplots(filas, cols, figsize=(18, filas * 3.2))
    axes = axes.flatten()

    paleta = {"Permanencia": "#2ecc71",
              "Renuncia voluntaria": "#e74c3c",
              "Renuncia involuntaria": "#f39c12"}

    for i, col in enumerate(NUM_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, ax=axes[i],
                     palette=paleta, kde=True, multiple="stack",
                     alpha=0.7, legend=(i == 0))
        axes[i].set_title(col, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Apagar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    ruta = FIGURES_DIR / "02_distribucion_numericas.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_grafico_categoricas(df: pd.DataFrame) -> Path:
    """Barras apiladas de variables categóricas vs target."""
    n = len(CAT_FEATURES)
    cols = 2
    filas = (n + cols - 1) // cols

    fig, axes = plt.subplots(filas, cols, figsize=(16, filas * 3.5))
    axes = axes.flatten()

    orden = ["Permanencia", "Renuncia voluntaria", "Renuncia involuntaria"]
    colores = ["#2ecc71", "#e74c3c", "#f39c12"]

    for i, col in enumerate(CAT_FEATURES):
        tabla = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
        # Reordenar columnas
        tabla = tabla.reindex(columns=[c for c in orden if c in tabla.columns])
        tabla.plot(kind="bar", stacked=True, ax=axes[i],
                   color=colores[:len(tabla.columns)], width=0.75)
        axes[i].set_title(f"{col}", fontsize=10, fontweight="bold")
        axes[i].set_ylabel("% de empleados")
        axes[i].set_xlabel("")
        axes[i].legend(title="", fontsize=7, loc="upper right")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    ruta = FIGURES_DIR / "03_distribucion_categoricas.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_matriz_correlacion(df: pd.DataFrame) -> Path:
    """Matriz de correlación de variables numéricas."""
    plt.figure(figsize=(11, 9))
    corr = df[NUM_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, annot_kws={"size": 8})
    plt.title("Matriz de Correlación - Variables Numéricas",
              fontweight="bold", fontsize=12)
    plt.tight_layout()
    ruta = FIGURES_DIR / "04_correlacion.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_boxplots_por_target(df: pd.DataFrame) -> Path:
    """Boxplots de variables numéricas clave por target."""
    # Seleccionamos las más relevantes para el negocio
    vars_clave = [
        "Salario_Mensual_HNL", "Antiguedad_Meses", "Evaluacion_Desempeno_1a5",
        "Horas_Extra_Mes", "Promociones", "Ausencias_Anio",
        "Dias_Vacaciones_Pendientes", "Carga_Trabajo_1a5",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(vars_clave):
        sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i],
                    palette=["#2ecc71", "#e74c3c", "#f39c12"])
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    ruta = FIGURES_DIR / "05_boxplots_target.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


# ============================================================
# 3. EJECUTOR PRINCIPAL DEL EDA
# ============================================================
def ejecutar_eda(df: pd.DataFrame) -> None:
    """Ejecuta el EDA completo y muestra resultados en consola."""
    print("\n" + "=" * 70)
    print(" ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 70)

    # 1. Resumen estadístico
    print("\n ESTADÍSTICAS DESCRIPTIVAS (Variables Numéricas):")
    print("-" * 70)
    print(resumen_estadistico(df).to_string())

    # 2. Distribución del target
    print("\n DISTRIBUCIÓN DEL TARGET:")
    print("-" * 70)
    print(analisis_target(df).to_string())

    # 3. Tasas de rotación por firma
    print("\n TASA DE ROTACIÓN POR FIRMA (%):")
    print("-" * 70)
    print(tasas_por_categoria(df, "Firma_Auditora").to_string())

    # 4. Área funcional
    print("\n TASA DE ROTACIÓN POR ÁREA FUNCIONAL (%):")
    print("-" * 70)
    print(tasas_por_categoria(df, "Area_Funcional").to_string())

    # 5. Nivel jerárquico
    print("\n TASA DE ROTACIÓN POR NIVEL JERÁRQUICO (%):")
    print("-" * 70)
    print(tasas_por_categoria(df, "Nivel_Jerarquico").to_string())

    # 6. Historial disciplinario
    print("\n  TASA DE ROTACIÓN POR HISTORIAL DISCIPLINARIO (%):")
    print("-" * 70)
    print(tasas_por_categoria(df, "Historial_Disciplinario").to_string())

    # 7. Programa de desarrollo
    print("\n TASA DE ROTACIÓN POR PROGRAMA DE DESARROLLO (%):")
    print("-" * 70)
    print(tasas_por_categoria(df, "Programa_Desarrollo").to_string())

    # 8. Generación de gráficos
    print("\n Generando gráficos...")
    print("-" * 70)

    ruta1 = guardar_grafico_target(df)
    print(f"   {ruta1.name}")

    ruta2 = guardar_grafico_numericas(df)
    print(f"   {ruta2.name}")

    ruta3 = guardar_grafico_categoricas(df)
    print(f"   {ruta3.name}")

    ruta4 = guardar_matriz_correlacion(df)
    print(f"   {ruta4.name}")

    ruta5 = guardar_boxplots_por_target(df)
    print(f"   {ruta5.name}")

    print(f"\n Todas las figuras en: {FIGURES_DIR}")
    print("\n EDA completado exitosamente")
    print("=" * 70)