import json
import re
from ollama import Client
from pydantic import ValidationError
from src.config import Config
from src.models.schemas import InvoiceExtraction
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Eres un experto en extraer datos de tickets y facturas de Malasia.

REGLA CRÍTICA: SOLO extrae datos que VES EXPLÍCITAMENTE en el texto.
SI NO VES UN DATO CLARAMENTE, devuelve null.
NUNCA inventes, adivines o supongas datos que no aparecen en el texto.

Debes devolver ÚNICAMENTE un objeto JSON válido con exactamente estos 4 campos:
- fecha: string con formato DD/MM/YYYY o null si NO EXISTE en el texto
- importe_total: número decimal (sin RM, sin comas) o null si NO EXISTE
- emisor: string con el nombre del comercio o null si NO EXISTE
- concepto: string breve o null si NO puedes determinar

EJEMPLO EXACTO DEL FORMATO:
{
  "fecha": "15/01/2019",
  "importe_total": 8.50,
  "emisor": "McDonald's",
  "concepto": "Compra general"
}

EJEMPLO SI NO HAY FECHA:
{
  "fecha": null,
  "importe_total": 8.50,
  "emisor": "McDonald's",
  "concepto": "Compra general"
}

INSTRUCCIONES ESPECÍFICAS:

1. FECHA - SOLO SI VES CLARAMENTE:
   - Patrón DD/MM/YYYY (ej: "15/01/2019", "09/02/2018")
   - Patrón DD-MM-YY (ej: "20-11-17") → convertir a DD/MM/2017
   - Patrón DD.MM.YY (ej: "23.03.18") → convertir a DD/MM/2018
   - Cerca de palabras: "Date:", "Bill Date:", "Invoice Date:"
   - SI NO VES NINGUNA FECHA → devuelve null (NO inventes)

2. IMPORTE TOTAL - Busca en este orden:
   - "Total (RM):" seguido de número
   - "TOTAL:" "Total Amount" seguido de número
   - "Amount Due" "Payment:" seguido de número
   - IGNORA: "CASH", "CHANGE", "Subtotal", "Rounding"
   - SI NO VES TOTAL CLARO → devuelve null

3. EMISOR - Primera línea en mayúsculas del ticket (nombre del negocio)

4. CONCEPTO - Descripción genérica: "Compra general", "Comida", "Servicios"

IMPORTANTE: Prefiere devolver null que inventar datos incorrectos.

RESPONDE SOLO CON EL JSON (sin ```json, sin markdown, sin explicaciones):
"""

USER_PROMPT_TEMPLATE = """Analiza este texto OCR de un ticket/factura de Malasia:

\"\"\"
{ocr_text}
\"\"\"

Devuelve EXACTAMENTE este formato JSON:
{{
  "fecha": "DD/MM/YYYY",
  "importe_total": 123.45,
  "emisor": "Nombre del comercio",
  "concepto": "Tipo de compra"
}}

Responde SOLO con el JSON (sin ```json, sin markdown, sin explicaciones):
"""

def call_ollama(ocr_text: str) -> dict:
    client = Client(host=Config.OLLAMA_HOST)

    logger.info(f"Llamando a Ollama modelo={Config.OLLAMA_MODEL}")

    try:
        # Nota: Si el texto es muy largo, considerar limitar con ocr_text[:8000]
        # para evitar exceder el limite de contexto del modelo
        response = client.chat(
            model=Config.OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
                }
            ],
            options={
                'temperature': 0.1  # Determinista para extraccion consistente de datos
            }
        )

        resp_text = response['message']['content']

        logger.info("="*80)
        logger.info("RESPUESTA COMPLETA DE LA IA:")
        logger.info(resp_text)
        logger.info("="*80)

        try:
            # Limpiar markdown
            resp_text = re.sub(r'```json\s*', '', resp_text)
            resp_text = re.sub(r'```\s*', '', resp_text)

            start = resp_text.find("{")
            end = resp_text.rfind("}") + 1

            if start == -1 or end == 0:
                logger.warning("No se encontró JSON en la respuesta")
                return {}

            json_extracted = resp_text[start:end]
            logger.info(f"JSON EXTRAIDO: {json_extracted}")

            parsed = json.loads(json_extracted)
            logger.info(f"JSON PARSEADO: {parsed}")

            # Validar con Pydantic
            try:
                invoice = InvoiceExtraction(**parsed)
                validated_dict = invoice.model_dump()
                logger.info(f"✓ VALIDACIÓN EXITOSA: {validated_dict}")
                return validated_dict
            except ValidationError as ve:
                logger.error(f"✗ VALIDACIÓN FALLIDA: {ve}")
                logger.error(f"Datos recibidos: {parsed}")
                return {}

        except json.JSONDecodeError as e:
            logger.warning(f"Error al parsear JSON. Error: {e}")
            logger.warning(f"Texto recibido: {resp_text[:500]}")
            return {}

    except Exception as e:
        logger.error(f"Error al llamar a Ollama: {e}")
        return {}
