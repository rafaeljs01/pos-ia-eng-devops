"""
Testes End-to-End (E2E) para a API.

Requerem o banco de dados rodando e a aplicação acessível.
Utilizamos o TestClient do FastAPI para simular requisições HTTP
no ambiente isolado (dentro do container).
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    """
    Testa se o endpoint de health check está respondendo corretamente.
    Valida a integração básica da API.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_root_endpoint(client):
    """
    Testa se o endpoint raiz retorna as informações da API.
    """
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "Running"
    assert "empresas" in data.get("endpoints", {})


def test_admin_sync_dashboard_empty_db(client):
    """
    Testa o endpoint do dashboard admin.
    Mesmo com banco vazio ou parcialmente populado, deve retornar estrutura correta.
    """
    response = client.get("/admin/sync")
    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert "total_months" in data["summary"]
    assert "months" in data
