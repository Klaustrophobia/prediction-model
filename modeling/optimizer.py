import time
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, CV_FOLDS, N_TRIALS_OPTUNA, TARGET_INV_MAP

# Silenciar logs de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 1. OPTIMIZADOR XGBOOST
# ============================================================
def optimizar_xgboost(X_train, y_train, n_trials: int = N_TRIALS_OPTUNA, verbose: bool = True):
    """
    Optimiza hiperparametros de XGBoost con Optuna.
    Metrica objetivo: ROC-AUC macro (validacion cruzada estratificada).
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            XGBClassifier(**params),
            X_train, y_train,
            cv=cv, scoring="roc_auc_ovr_weighted", n_jobs=-1,
        )
        return scores.mean()

    if verbose:
        print("\n  Iniciando optimizacion de XGBoost...")
        print(f"  Trials: {n_trials}, Folds CV: {CV_FOLDS}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if verbose:
        print(f"  Mejor ROC-AUC (CV): {study.best_value:.4f}")
        print(f"  Mejores parametros:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"    {k:<20} {v:.6f}")
            else:
                print(f"    {k:<20} {v}")

    return study.best_params, study.best_value


# ============================================================
# 2. OPTIMIZADOR LIGHTGBM
# ============================================================
def optimizar_lightgbm(X_train, y_train, n_trials: int = N_TRIALS_OPTUNA, verbose: bool = True):
    """
    Optimiza hiperparametros de LightGBM con Optuna.
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            LGBMClassifier(**params),
            X_train, y_train,
            cv=cv, scoring="roc_auc_ovr_weighted", n_jobs=-1,
        )
        return scores.mean()

    if verbose:
        print("\n  Iniciando optimizacion de LightGBM...")
        print(f"  Trials: {n_trials}, Folds CV: {CV_FOLDS}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if verbose:
        print(f"  Mejor ROC-AUC (CV): {study.best_value:.4f}")
        print(f"  Mejores parametros:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"    {k:<20} {v:.6f}")
            else:
                print(f"    {k:<20} {v}")

    return study.best_params, study.best_value


# ============================================================
# 3. EVALUACION FINAL EN TEST
# ============================================================
def evaluar_en_test(modelo, X_test, y_test, nombre: str = "Modelo"):
    """
    Evaluacion final sobre el conjunto de test (holdout).
    Muestra metricas, reporte de clasificacion y matriz de confusion.
    """
    t0 = time.time()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    t_pred = time.time() - t0

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
    }

    print("\n" + "=" * 78)
    print(f"EVALUACION FINAL EN TEST - {nombre}")
    print("=" * 78)
    print(f"  Registros de test: {X_test.shape[0]}")
    print(f"  Tiempo de prediccion: {t_pred:.4f} s\n")

    print("  Metricas globales:")
    print("  " + "-" * 60)
    for k, v in metricas.items():
        print(f"    {k:<20} {v:.4f}")

    clases = [TARGET_INV_MAP[i] for i in range(len(TARGET_INV_MAP))]

    print("\n  Reporte de clasificacion:")
    print("  " + "-" * 70)
    print(classification_report(y_test, y_pred, target_names=clases, digits=4, zero_division=0))

    print("  Matriz de confusion:")
    print("  " + "-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    header = "  " + " " * 30 + "  ".join(f"{c[:12]:>12}" for c in clases)
    print(header)
    for i, fila in enumerate(cm):
        valores = "  ".join(f"{v:>12}" for v in fila)
        print(f"  {clases[i][:28]:<30}{valores}")

    return metricas, y_pred, y_proba


# ============================================================
# 4. WRAPPER PRINCIPAL
# ============================================================
def optimizar_modelo(
    nombre_modelo: str,
    X_train, y_train,
    n_trials: int = N_TRIALS_OPTUNA,
    verbose: bool = True,
):
    """
    Optimiza el modelo indicado y devuelve instancia ya entrenada.
    Retorna (modelo_entrenado, mejores_params, mejor_score_cv).
    """
    if nombre_modelo == "XGBoost":
        params, score = optimizar_xgboost(X_train, y_train, n_trials, verbose)
        params.update({
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        })
        modelo = XGBClassifier(**params)
    elif nombre_modelo == "LightGBM":
        params, score = optimizar_lightgbm(X_train, y_train, n_trials, verbose)
        params.update({
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        })
        modelo = LGBMClassifier(**params)
    else:
        raise ValueError(f"Modelo no soportado para optimizacion: {nombre_modelo}")

    if verbose:
        print(f"\n  Entrenando modelo final con los mejores parametros...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    t_fit = time.time() - t0
    if verbose:
        print(f"  Entrenamiento completado en {t_fit:.2f} s")

    return modelo, params, score


if __name__ == "__main__":
    from data_pipeline.data_loader import cargar_datos
    from data_pipeline.preprocessing import preparar_datos

    df = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preparar_datos(df, verbose=False)

    modelo, params, score = optimizar_modelo("XGBoost", X_train, y_train, n_trials=10)
    evaluar_en_test(modelo, X_test, y_test, "XGBoost optimizado")