from pydantic import BaseModel, Field


class EmpleadoInput(BaseModel):
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


class PrediccionOutput(BaseModel):
    id_empleado: str | None = None
    prob_permanencia: float
    prob_renuncia_voluntaria: float
    prob_renuncia_involuntaria: float
    nivel_riesgo: str
    factores_riesgo: list[dict]
    factores_proteccion: list[dict]
    recomendaciones: list[str]


class BatchInput(BaseModel):
    empleados: list[EmpleadoInput]


class BatchOutput(BaseModel):
    total: int
    predicciones: list[PrediccionOutput]


<<<<<<< HEAD
=======

>>>>>>> 55c495b (Cleaning stage)
class ModelInfo(BaseModel):
    nombre: str
    version: str
    tipo: str
    metricas: dict
    features: int
<<<<<<< HEAD
    clases: list[str]
    fecha_entrenamiento: str | None = None
=======
    clases: List[str]
    fecha_entrenamiento: Optional[str] = None


class FeatureImportance(BaseModel):
    """Importancia de una feature segun SHAP."""
    rank: int
    feature: str
    importancia: float
>>>>>>> 55c495b (Cleaning stage)


class FeatureImportanceOutput(BaseModel):
    clase: str
    top_features: list[dict]

<<<<<<< HEAD

=======
>>>>>>> 55c495b (Cleaning stage)
class HealthCheck(BaseModel):
    status: str
    modelo_cargado: bool
    preprocessor_cargado: bool
    timestamp: str
