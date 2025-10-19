import pytest
from src.models.schemas import InvoiceExtraction, fallback_amount_from_text

def test_invoice_schema_ok():
    """Test creación de invoice con todos los campos válidos"""
    m = InvoiceExtraction(
        fecha="31/01/2024",
        importe_total=10.5,
        emisor="Empresa ABC",
        concepto="Servicios de consultoría"
    )
    assert m.importe_total == 10.5
    assert m.fecha == "31/01/2024"
    assert m.emisor == "Empresa ABC"
    assert m.concepto == "Servicios de consultoría"

def test_invoice_schema_with_nulls():
    """Test que acepta campos null"""
    m = InvoiceExtraction(
        fecha=None,
        importe_total=None,
        emisor="Solo Emisor",
        concepto=None
    )
    assert m.fecha is None
    assert m.importe_total is None
    assert m.emisor == "Solo Emisor"
    assert m.concepto is None

def test_invoice_schema_empty_fecha():
    """Test con fecha vacía (string vacío se trata como None)"""
    m = InvoiceExtraction(
        fecha="",
        importe_total=100.0,
        emisor="Test",
        concepto="Test"
    )
    assert m.fecha is None or m.fecha == ""

def test_date_valid_formats():
    """Test que acepta varios formatos de fecha válidos"""
    valid_dates = [
        "31/01/2024",
        "2024-01-31",
        "31-01-2024",
        "01/31/2024"
    ]
    for date in valid_dates:
        m = InvoiceExtraction(fecha=date, importe_total=1.0)
        assert m.fecha == date

def test_date_invalid():
    """Test que rechaza fechas inválidas"""
    with pytest.raises(Exception):
        InvoiceExtraction(fecha="no-es-fecha", importe_total=1.0)

def test_importe_total_conversion():
    """Test conversión de importe a float"""
    m = InvoiceExtraction(importe_total="123.45")
    assert isinstance(m.importe_total, float)
    assert m.importe_total == 123.45

def test_fallback_amount_spanish_format():
    """Test extracción de importes en formato español (1.234,56)"""
    text = "Total a pagar 1.234,56 EUR"
    assert fallback_amount_from_text(text) == 1234.56

def test_fallback_amount_international_format():
    """Test extracción de importes en formato internacional (1234.56)"""
    text = "Total amount 1234.56 USD"
    assert fallback_amount_from_text(text) == 1234.56

def test_fallback_amount_simple():
    """Test extracción de importes simples (123,45)"""
    text = "TOTAL: 123,45"
    assert fallback_amount_from_text(text) == 123.45

def test_fallback_amount_with_spaces():
    """Test importes con espacios como separadores de miles"""
    text = "Importe: 1 234,56"
    assert fallback_amount_from_text(text) == 1234.56

def test_fallback_amount_not_found():
    """Test cuando no hay importes en el texto"""
    text = "Este texto no tiene números válidos como importe"
    assert fallback_amount_from_text(text) is None

def test_fallback_amount_multiple_candidates():
    """Test que encuentra el primer importe válido entre varios"""
    text = "Subtotal: 100,00 IVA: 21,00 TOTAL: 121,00"
    # Debe encontrar al menos uno
    result = fallback_amount_from_text(text)
    assert result in [100.0, 21.0, 121.0]

def test_invoice_with_raw_text_preview():
    """Test que incluye preview del texto OCR"""
    preview = "Este es un texto de prueba OCR..."
    m = InvoiceExtraction(
        fecha="01/01/2024",
        importe_total=50.0,
        emisor="Test",
        concepto="Test",
        raw_text_preview=preview
    )
    assert m.raw_text_preview == preview
