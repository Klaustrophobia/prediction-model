from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ============================================================
# 1. ENTRADA: Datos de un empleado
# ============================================================
class EmpleadoInput(BaseModel):
    """Datos de un empleado para prediccion."""

    # Identificacion (opcional en prediccion)
    ID_Empleado: Optional[str] = Field(None, description="ID del empleado")

    # Categoricas
    Firma_Auditora: str = Field(..., description="Deloitte, KPMG o PwC")
    Genero: str = Field(..., description="Masculino o Femenino")
    Estado_Civil: str = Field(..., description="Soltero/a, Casado/a, Union libre, Divorciado/a")
    Nivel_Educativo: str = Field(..., description="Licenciatura, Maestria, Doctorado")
    Universidad_Procedencia: str = Field(..., description="UNITEC, UNAH, UNICAH, etc.")
    Cargo: str = Field(..., description="Cargo del empleado")
    Area_Funcional: str = Field(..., description="Auditoria, Impuestos, Consultoria, Apoyo administrativo")
    Nivel_Jerarquico: str = Field(..., description="Operativo, Profesional, Mando medio, Gerencial, Directivo")
    Programa_Desarrollo: str = Field(..., description="Si o No")
    Historial_Disciplinario: str = Field("Ninguno", description="Ninguno, Llamado de atencion, Amonestacion escrita")

    # Numericas
    Edad: int = Field(..., ge=18, le=70, description="Edad en anios")
    Antiguedad_Meses: float = Field(..., ge=0, description="Antiguedad en meses")
    Salario_Mensual_HNL: float = Field(..., ge=0, description="Salario mensual en HNL")
    Incremento_Salarial_Porcentaje: float = Field(..., ge=0, description="Incremento salarial %")
    Evaluacion_Desempeno_1a5: float = Field(..., ge=1, le=5, description="Evaluacion de desempeno")
    Horas_Extra_Mes: float = Field(..., ge=0, description="Horas extra por mes")
    Horas_Capacitacion_Anio: float = Field(..., ge=0, description="Horas de capacitacion por anio")
    Promociones: int = Field(..., ge=0, description="Numero de promociones")
    Ausencias_Anio: int = Field(..., ge=0, description="Ausencias en el anio")
    Dias_Vacaciones_Pendientes: int = Field(..., ge=0, description="Dias de vacaciones pendientes")
    Distancia_Domicilio_Trabajo_km: float = Field(..., ge=0, description="Distancia en km")
    Tiempo_Desplazamiento_Min: float = Field(..., ge=0, description="Tiempo de desplazamiento en minutos")
    Carga_Trabajo_1a5: float = Field(..., ge=1, le=5, description="Carga de trabajo 1-5")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ID_Empleado": "HN-AUD-0001",
                "Firma_Auditora": "KPMG",
                "Genero": "Femenino",
                "Estado_Civil": "Soltero/a",
                "Nivel_Educativo": "Licenciatura",
                "Universidad_Procedencia": "UNITEC",
                "Cargo": "Asistente de Impuestos",
                "Area_Funcional": "Impuestos",
                "Nivel_Jerarquico": "Operativo",
                "Programa_Desarrollo": "Si",
                "Historial_Disciplinario": "Ninguno",
                "Edad": 43,
                "Antiguedad_Meses": 63,
                "Salario_Mensual_HNL": 27000,
                "Incremento_Salarial_Porcentaje": 5,
                "Evaluacion_Desempeno_1a5": 4,
                "Horas_Extra_Mes": 2.2,
                "Horas_Capacitacion_Anio": 29,
                "Promociones": 3,
                "Ausencias_Anio": 3,
                "Dias_Vacaciones_Pendientes": 0,
                "Distancia_Domicilio_Trabajo_km": 4.2,
                "Tiempo_Desplazamiento_Min": 21,
                "Carga_Trabajo_1a5": 2,
            }
        }
    }


# ============================================================
# 2. SALIDA: Prediccion de un empleado
# ============================================================
class FactorSHAP(BaseModel):
    """Un factor individual con su contribucion SHAP."""
    feature: str
    shap: float
    valor: float


class PrediccionOutput(BaseModel):
    """Respuesta de prediccion individual."""
    ID_Empleado: Optional[str] = None
    probabilidad_permanencia: float
    probabilidad_renuncia_voluntaria: float
    probabilidad_renuncia_involuntaria: float
    nivel_riesgo: str  # Bajo, Moderado, Alto
    prediccion: str  # Clase predicha
    factores_riesgo: List[FactorSHAP]
    factores_proteccion: List[FactorSHAP]
    recomendaciones: List[str]


# ============================================================
# 3. ENTRADA/SALIDA: Batch
# ============================================================
class BatchInput(BaseModel):
    """Lista de empleados para prediccion masiva."""
    empleados: List[EmpleadoInput]


class BatchOutput(BaseModel):
    """Respuesta con predicciones de todos los empleados."""
    total: int
    predicciones: List[PrediccionOutput]


# ============================================================
# 4. SALIDA: Info del modelo
# ============================================================
class ModelInfo(BaseModel):
    """Informacion del modelo cargado."""
    nombre: str
    version: str
    tipo: str
    metricas: Dict[str, float]
    features: int
    clases: List[str]
    fecha_entrenamiento: Optional[str] = None


# ============================================================
# 5. SALIDA: Importancia de features
# ============================================================
class FeatureImportance(BaseModel):
    """Importancia de una feature segun SHAP."""
    rank: int
    feature: str
    importancia: float


class FeatureImportanceOutput(BaseModel):
    """Importancia global de features para una clase."""
    clase: str
    top_features: List[FeatureImportance]


# ============================================================
# 6. HEALTH CHECK
# ============================================================
class HealthCheck(BaseModel):
    """Estado del servicio."""
    status: str
    modelo_cargado: bool
    preprocessor_cargado: bool
    timestamp: str