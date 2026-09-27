import pandas as pd
from pathlib import Path
from config import DATA_FILE, SHEET_NAME, TARGET


def cargar_datos(ruta: Path = DATA_FILE, hoja: str = SHEET_NAME,
                 verbose: bool = True) -> pd.DataFrame:

    if not ruta.exists():
        raise FileNotFoundError(
            f"\nNo se encontro el archivo: {ruta}\n"
            f"   Por favor, coloca el Excel en: {ruta.parent}\n"
            f"   Nombre esperado: {ruta.name}\n"
        )

    if verbose:
        print(f"Cargando datos desde: {ruta.name}")
        print(f"   Ruta completa: {ruta}")

    df = pd.read_excel(ruta, sheet_name=hoja)

    if verbose:
        print(f"Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")

    return df


def validar_datos(df: pd.DataFrame) -> dict:
    resumen = {
        "n_registros": len(df),
        "n_columnas": df.shape[1],
        "ids_unicos": df["ID_Empleado"].nunique(),
        "valores_nulos_total": int(df.isnull().sum().sum()),
        "columnas_con_nulos": df.columns[df.isnull().any()].tolist(),
        "distribucion_target": df[TARGET].value_counts().to_dict(),
    }
    return resumen


def imprimir_resumen(resumen: dict) -> None:
    print("\n" + "=" * 65)
    print(" RESUMEN DE VALIDACIÓN DE DATOS")
    print("=" * 65)
    print(f"  Registros totales:        {resumen['n_registros']}")
    print(f"  Columnas:                 {resumen['n_columnas']}")
    print(f"  IDs únicos:               {resumen['ids_unicos']}")
    print(f"  Valores nulos totales:    {resumen['valores_nulos_total']}")

    if resumen['columnas_con_nulos']:
        print(f"  Columnas con nulos:       {resumen['columnas_con_nulos']}")
    else:
        print("  Columnas con nulos:       Ninguna ✓")

    print(f"\n  Distribución de '{TARGET}':")
    for k, v in resumen["distribucion_target"].items():
        pct = v / resumen["n_registros"] * 100
        print(f"     • {k:<25} {v:>4} ({pct:>5.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    df = cargar_datos()
    resumen = validar_datos(df)
    imprimir_resumen(resumen)
    print("\n🔍 Primeras 5 filas:")
    print(df.head().to_string())
