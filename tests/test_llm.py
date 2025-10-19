from unittest.mock import patch, MagicMock
from src.llm.ollama_llm import call_ollama

@patch("src.llm.ollama_llm.Client")
def test_call_ollama_parses_json(mock_client_class):
    # Mock del cliente Ollama
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simula respuesta exitosa del chat
    mock_client.chat.return_value = {
        'message': {
            'content': '{"fecha":"2024-01-31","importe_total":99.9,"emisor":"Demo","concepto":"Compra"}'
        }
    }

    out = call_ollama("texto de prueba")

    assert out["emisor"] == "Demo"
    assert out["importe_total"] == 99.9
    assert out["fecha"] == "2024-01-31"
    assert out["concepto"] == "Compra"

@patch("src.llm.ollama_llm.Client")
def test_call_ollama_invalid_json(mock_client_class):
    # Mock del cliente cuando devuelve texto sin JSON
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.chat.return_value = {
        'message': {
            'content': 'No puedo extraer datos de este documento'
        }
    }

    out = call_ollama("texto inválido")

    # Debe devolver diccionario vacío cuando no hay JSON válido
    assert out == {}

@patch("src.llm.ollama_llm.Client")
def test_call_ollama_partial_json(mock_client_class):
    # Mock cuando el modelo devuelve JSON con algunos campos null
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.chat.return_value = {
        'message': {
            'content': '{"fecha":null,"importe_total":150.0,"emisor":"Empresa X","concepto":null}'
        }
    }

    out = call_ollama("factura incompleta")

    assert out["emisor"] == "Empresa X"
    assert out["importe_total"] == 150.0
    assert out["fecha"] is None
    assert out["concepto"] is None

@patch("src.llm.ollama_llm.Client")
def test_call_ollama_connection_error(mock_client_class):
    # Mock cuando hay error de conexión
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.chat.side_effect = Exception("Connection refused")

    out = call_ollama("texto")

    # Debe devolver diccionario vacío en caso de error
    assert out == {}
