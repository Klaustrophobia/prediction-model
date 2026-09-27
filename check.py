import sys
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import matplotlib
import seaborn as sns

print(f" Python: {sys.version}")
print(f" Ejecutable: {sys.executable}")
print(f" pandas: {pd.__version__}")
print(f" numpy: {np.__version__}")
print(f" scikit-learn: {sklearn.__version__}")
print(f" xgboost: {xgb.__version__}")
print(f" lightgbm: {lgb.__version__}")
print(f" optuna: {optuna.__version__}")
print(f" shap: {shap.__version__}")
print(f" matplotlib: {matplotlib.__version__}")
print(f" seaborn: {sns.__version__}")
print("\n🎉 Todas las dependencias están instaladas correctamente")
