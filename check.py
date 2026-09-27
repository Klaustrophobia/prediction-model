"""Verificación de dependencias - Windows."""
import sys

print(f" Python: {sys.version}")
print(f" Ejecutable: {sys.executable}")

import pandas as pd
print(f" pandas: {pd.__version__}")

import numpy as np
print(f" numpy: {np.__version__}")

import sklearn
print(f" scikit-learn: {sklearn.__version__}")

import xgboost as xgb
print(f" xgboost: {xgb.__version__}")

import lightgbm as lgb
print(f" lightgbm: {lgb.__version__}")

import optuna
print(f" optuna: {optuna.__version__}")

import shap
print(f" shap: {shap.__version__}")

import matplotlib
print(f" matplotlib: {matplotlib.__version__}")

import seaborn as sns
print(f" seaborn: {sns.__version__}")

print("\n🎉 Todas las dependencias están instaladas correctamente")