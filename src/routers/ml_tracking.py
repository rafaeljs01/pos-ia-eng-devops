import os
import logging
from typing import Dict, Any, List, Optional
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml/tracking",
    tags=["MLflow Tracking"]
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Instancia o cliente do MLflow
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# --- Schemas ---

class ExperimentCreate(BaseModel):
    name: str
    artifact_location: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

class RunCreate(BaseModel):
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

class ParamLog(BaseModel):
    key: str
    value: str

class MetricLog(BaseModel):
    key: str
    value: float
    step: Optional[int] = 0

class TagLog(BaseModel):
    key: str
    value: str

class RunStatusUpdate(BaseModel):
    status: str  # ex: FINISHED, FAILED, KILLED


# --- Endpoints: Experiments ---

@router.post("/experiments", status_code=201)
async def create_experiment(request: ExperimentCreate):
    """
    Cria um novo experimento no MLflow.
    """
    try:
        experiment_id = client.create_experiment(
            name=request.name,
            artifact_location=request.artifact_location,
            tags=request.tags
        )
        return {"experiment_id": experiment_id, "message": "Experimento criado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao criar experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments")
async def list_experiments(max_results: int = 100):
    """
    Lista os experimentos existentes no MLflow.
    """
    try:
        experiments = client.search_experiments(max_results=max_results)
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": exp.tags,
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Erro ao buscar experimentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Endpoints: Runs ---

@router.post("/experiments/{experiment_id}/runs", status_code=201)
async def create_run(experiment_id: str, request: RunCreate):
    """
    Inicia uma nova chamada de teste (Run) vinculada a um experimento.
    """
    try:
        run = client.create_run(
            experiment_id=experiment_id,
            run_name=request.run_name,
            tags=request.tags
        )
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "message": "Run criado com sucesso."
        }
    except Exception as e:
        logger.error(f"Erro ao criar run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """
    Recupera as características atuais de uma chamada de teste.
    """
    try:
        run = client.get_run(run_id)
        return {
            "info": {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            },
            "data": {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        }
    except Exception as e:
        logger.error(f"Erro ao buscar o run {run_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Run {run_id} não encontrado ou erro interno.")

@router.post("/runs/{run_id}/log-param")
async def log_param(run_id: str, request: ParamLog):
    """
    Registra um parâmetro na chamada de teste (ex: hiperparâmetro).
    """
    try:
        client.log_param(run_id, request.key, request.value)
        return {"message": f"Parâmetro '{request.key}' registrado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao registrar parâmetro no run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/runs/{run_id}/log-metric")
async def log_metric(run_id: str, request: MetricLog):
    """
    Registra uma métrica na chamada de teste (ex: accuracy, rmse).
    """
    try:
        client.log_metric(
            run_id=run_id,
            key=request.key,
            value=request.value,
            step=request.step,
            timestamp=int(time.time() * 1000)
        )
        return {"message": f"Métrica '{request.key}' registrada com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao registrar métrica no run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/runs/{run_id}/log-tag")
async def log_tag(run_id: str, request: TagLog):
    """
    Adiciona uma tag para facilitar buscas e identificação da chamada de teste.
    """
    try:
        client.set_tag(run_id, request.key, request.value)
        return {"message": f"Tag '{request.key}' registrada com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao registrar tag no run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/runs/{run_id}/status")
async def update_run_status(run_id: str, request: RunStatusUpdate):
    """
    Atualiza o status da chamada (ex: FINISHED, FAILED, KILLED).
    """
    valid_statuses = ["FINISHED", "FAILED", "KILLED"]
    if request.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status inválido. Permitidos: {valid_statuses}"
        )
    
    try:
        client.set_terminated(run_id, status=request.status, end_time=int(time.time() * 1000))
        return {"message": f"Status do run {run_id} atualizado para {request.status}."}
    except Exception as e:
        logger.error(f"Erro ao atualizar status do run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
