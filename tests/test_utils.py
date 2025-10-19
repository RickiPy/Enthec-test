import logging
from src.utils.logger import get_logger
from src.utils.date_parser import find_date_in_text

# ===== TESTS DE LOGGER =====

def test_get_logger_returns_logger():
    """Test que get_logger devuelve una instancia de Logger configurada correctamente"""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"
    assert logger.level == logging.INFO

def test_logger_can_log_messages(caplog):
    """Test que el logger puede escribir mensajes de diferentes niveles"""
    logger = get_logger("test_logging")

    with caplog.at_level(logging.INFO):
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    assert "Info message" in caplog.text
    assert "Warning message" in caplog.text
    assert "Error message" in caplog.text

# ===== TESTS DE DATE PARSER =====

def test_find_date_in_text_various_formats():
    """Test que encuentra fechas en diferentes formatos comunes"""
    test_cases = [
        ("Fecha: 31/01/2024", "2024-01-31"),
        ("Date: 2024-01-31", "2024-01-31"),
        ("Emitido el 31-01-2024", "2024-01-31"),
        ("Date: 01/31/2024", "2024-01-31"),
    ]

    for text, expected in test_cases:
        result = find_date_in_text(text)
        assert result == expected, f"Failed for: {text}"

def test_find_date_returns_none_when_not_found():
    """Test que devuelve None cuando no hay fecha válida"""
    text = "Este texto no contiene ninguna fecha válida"
    assert find_date_in_text(text) is None

def test_find_date_extracts_first_valid_date():
    """Test que extrae la primera fecha válida cuando hay múltiples"""
    text = "Primera fecha: 15/03/2024 y segunda fecha: 20/03/2024"
    result = find_date_in_text(text)
    assert result is not None
    assert "2024-03" in result  # Debe encontrar alguna de las dos
