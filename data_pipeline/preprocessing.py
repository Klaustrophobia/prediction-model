"""
Preprocesamiento y Feature Engineering.
Ubicacion: data_pipeline/preprocessing.py

Compatible con Windows / Linux / macOS
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from config import (
    CAT_FEATURES, NUM_FEATURES, TARGET, COLS_EXCLUIR,
    TARGET_MAP, RANDOM_STATE, TEST_SIZE, VAL_SIZE, PREPROCESSOR_FILE,
)


# ============================================================
# 1. FEATURE ENGINEER
# ============================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea features derivadas especificas para firmas auditoras.
    Recibe un DataFrame con las columnas originales y agrega nuevas.
    """

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer espera un pandas DataFrame")

        df = X.copy()

        # -----------------------------------------------------
        # 1. Ratio salario / antiguedad
        #    Mide si el salario crece con la antiguedad
        # -----------------------------------------------------
        df["Ratio_Salario_Antiguedad"] = (
            df["Salario_Mensual_HNL"] / (df["Antiguedad_Meses"] + 1)
        )

        # -----------------------------------------------------
        # 2. Score de burnout
        #    Combina horas extra, carga laboral y ausencias
        # -----------------------------------------------------
        df["Score_Burnout"] = (
            (df["Horas_Extra_Mes"] / 40.0) * 0.5
            + (df["Carga_Trabajo_1a5"] / 5.0) * 0.3
            + (df["Ausencias_Anio"] / 20.0) * 0.2
        ) * 10

        # -----------------------------------------------------
        # 3. Ratio capacitacion / antiguedad
        # -----------------------------------------------------
        df["Ratio_Capacitacion_Antiguedad"] = (
            df["Horas_Capacitacion_Anio"] / (df["Antiguedad_Meses"] + 1)
        )

        # -----------------------------------------------------
        # 4. Ratio promociones por anio
        # -----------------------------------------------------
        df["Ratio_Promocion_Anios"] = (
            df["Promociones"] / ((df["Antiguedad_Meses"] / 12) + 1)
        )

        # -----------------------------------------------------
        # 5. Score de estabilidad
        #    Mayor antiguedad + menor ausencia + mas promociones
        # -----------------------------------------------------
        df["Score_Estabilidad"] = (
            (df["Antiguedad_Meses"] / 84.0) * 0.5
            + (1 - (df["Ausencias_Anio"] / 20.0)) * 0.3
            + (df["Promociones"] / 4.0) * 0.2
        )

        # -----------------------------------------------------
        # 6. Score de insatisfaccion
        #    Vacaciones pendientes + ausencias + carga alta
        # -----------------------------------------------------
        df["Score_Insatisfaccion"] = (
            (df["Dias_Vacaciones_Pendientes"] / 30.0) * 0.5
            + (df["Ausencias_Anio"] / 20.0) * 0.3
            + (df["Carga_Trabajo_1a5"] / 5.0) * 0.2
        )

        # -----------------------------------------------------
        # 7. Carga neta
        #    Horas extra + carga - capacitacion
        # -----------------------------------------------------
        df["Carga_Neta"] = (
            df["Horas_Extra_Mes"] * (df["Carga_Trabajo_1a5"] / 5.0)
            - (df["Horas_Capacitacion_Anio"] / 12.0)
        )

        # -----------------------------------------------------
        # 8. Ratio ausencias por antiguedad
        # -----------------------------------------------------
        df["Ratio_Ausencias_Antiguedad"] = (
            df["Ausencias_Anio"] / ((df["Antiguedad_Meses"] / 12) + 1)
        )

        # -----------------------------------------------------
        # 9. Score de desarrollo
        #    Programa + capacitacion + promociones
        # -----------------------------------------------------
        programa_num = (df["Programa_Desarrollo"] == "Si").astype(int)
        df["Score_Desarrollo"] = (
            programa_num * 0.4
            + (df["Horas_Capacitacion_Anio"] / 100.0) * 0.4
            + (df["Promociones"] / 4.0) * 0.2
        )

        # -----------------------------------------------------
        # 10. Ratio distancia / tiempo de desplazamiento
        #     Indica si vive lejos o viaja mucho tiempo
        # -----------------------------------------------------
        df["Ratio_Distancia_Tiempo"] = (
            df["Distancia_Domicilio_Trabajo_km"] / (df["Tiempo_Desplazamiento_Min"] + 1)
        )

        self.feature_names_ = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


# ============================================================
# 2. DATA PREPROCESSOR
# ============================================================
class DataPreprocessor:
    """
    Pipeline completo: Feature Engineering + Imputacion + Encoding + Escalado.
    """

    def __init__(self):
        self.numeric_features = NUM_FEATURES.copy()
        self.categorical_features = CAT_FEATURES.copy()
        self.derived_features = [
            "Ratio_Salario_Antiguedad",
            "Score_Burnout",
            "Ratio_Capacitacion_Antiguedad",
            "Ratio_Promocion_Anios",
            "Score_Estabilidad",
            "Score_Insatisfaccion",
            "Carga_Neta",
            "Ratio_Ausencias_Antiguedad",
            "Score_Desarrollo",
            "Ratio_Distancia_Tiempo",
        ]

        self.feature_engineer = FeatureEngineer()
        self.preprocessor = None
        self.final_features_ = None
        self.label_encoders_ = {}

    # -----------------------------------------------------
    # Construir el ColumnTransformer
    # -----------------------------------------------------
    def _build_preprocessor(self):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        all_numeric = self.numeric_features + self.derived_features

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, all_numeric),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )
        return preprocessor

    # -----------------------------------------------------
    # Codificar categoricas con LabelEncoder (uno por columna)
    # -----------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.categorical_features:
            if col not in df_out.columns:
                continue
            if fit:
                le = LabelEncoder()
                df_out[col] = le.fit_transform(df_out[col].astype(str))
                self.label_encoders_[col] = le
            else:
                le = self.label_encoders_.get(col)
                if le is None:
                    raise ValueError(f"LabelEncoder no encontrado para columna: {col}")
                # Manejar categorias no vistas
                valores = df_out[col].astype(str)
                clases_conocidas = set(le.classes_)
                valores = valores.apply(lambda v: v if v in clases_conocidas else le.classes_[0])
                df_out[col] = le.transform(valores)
        return df_out

    # -----------------------------------------------------
    # Fit + Transform
    # -----------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        print("  [Preprocessor] Aplicando Feature Engineering...")
        df_eng = self.feature_engineer.fit_transform(df)

        print("  [Preprocessor] Codificando variables categoricas...")
        df_enc = self._encode_categoricals(df_eng, fit=True)

        print("  [Preprocessor] Ajustando pipeline (imputacion + escalado)...")
        self.preprocessor = self._build_preprocessor()
        X_out = self.preprocessor.fit_transform(df_enc)

        self.final_features_ = (
            self.numeric_features + self.derived_features + self.categorical_features
        )
        print(f"  [Preprocessor] Listo. Dimensiones: {X_out.shape}")
        return X_out

    # -----------------------------------------------------
    # Transform (para datos nuevos)
    # -----------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("El preprocesador no esta ajustado. Llama a fit_transform primero.")

        df_eng = self.feature_engineer.transform(df)
        df_enc = self._encode_categoricals(df_eng, fit=False)
        return self.preprocessor.transform(df_enc)

    # -----------------------------------------------------
    # Nombres finales de features
    # -----------------------------------------------------
    def get_feature_names(self):
        if self.final_features_ is None:
            raise ValueError("Primero llama a fit_transform.")
        return self.final_features_

    # -----------------------------------------------------
    # Guardar / Cargar
    # -----------------------------------------------------
    def save(self, ruta: Path = PREPROCESSOR_FILE):
        joblib.dump(self, ruta)
        print(f"  [Preprocessor] Guardado en: {ruta}")

    @staticmethod
    def load(ruta: Path = PREPROCESSOR_FILE):
        return joblib.load(ruta)


# ============================================================
# 3. DIVISION DE DATOS
# ============================================================
def preparar_datos(df: pd.DataFrame, verbose: bool = True):
    """
    Prepara los datos:
      1. Elimina columnas con data leakage
      2. Aplica preprocesamiento
      3. Divide en train / validation / test
    Retorna:
      X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    # 1. Eliminar columnas con data leakage y mapear target
    df_clean = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns]).copy()

    if TARGET not in df_clean.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no esta en los datos.")

    y = df_clean[TARGET].map(TARGET_MAP).values
    X = df_clean.drop(columns=[TARGET])

    # 2. Division train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Division train / val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    if verbose:
        print("\n  Division de datos:")
        print(f"    Train:      {X_train.shape[0]:>4} registros")
        print(f"    Validation: {X_val.shape[0]:>4} registros")
        print(f"    Test:       {X_test.shape[0]:>4} registros")

    # 4. Preprocesamiento
    preprocessor = DataPreprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    return X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    from data_pipeline.data_loader import cargar_datos

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, pp = preparar_datos(df)
    print("\nFormas finales:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  y_test:  {y_test.shape}")