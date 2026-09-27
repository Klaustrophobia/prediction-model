from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Base_de_Datos_Modelo.xlsx"
SHEET_NAME = "Base_Datos"

TARGET = "Rotacion_Personal"
TARGET_CLASSES = ["Permanencia", "Renuncia voluntaria", "Renuncia involuntaria"]

TARGET_MAP = {
    "Permanencia": 0,
    "Renuncia voluntaria": 1,
    "Renuncia involuntaria": 2,
}
TARGET_INV_MAP = {v: k for k, v in TARGET_MAP.items()}

COLS_EXCLUIR = [
    "ID_Empleado",
    "Fecha_Salida",
    "Estado_Colaborador",
    "Motivo_Salida",
    "Año_Observacion",
    "Fecha_Ingreso",
]

CAT_FEATURES = [
    "Firma_Auditora",
    "Genero",
    "Estado_Civil",
    "Nivel_Educativo",
    "Universidad_Procedencia",
    "Cargo",
    "Area_Funcional",
    "Nivel_Jerarquico",
    "Programa_Desarrollo",
    "Historial_Disciplinario",
]

NUM_FEATURES = [
    "Edad",
    "Antiguedad_Meses",
    "Salario_Mensual_HNL",
    "Incremento_Salarial_Porcentaje",
    "Evaluacion_Desempeno_1a5",
    "Horas_Extra_Mes",
    "Horas_Capacitacion_Anio",
    "Promociones",
    "Ausencias_Anio",
    "Dias_Vacaciones_Pendientes",
    "Distancia_Domicilio_Trabajo_km",
    "Tiempo_Desplazamiento_Min",
    "Carga_Trabajo_1a5",
]

DERIVED_FEATURES = [
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

PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
CV_FOLDS = 5
N_TRIALS_OPTUNA = 30

ALERT_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40
