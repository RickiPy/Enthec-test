from pathlib import Path
import json
import tempfile
import shutil
from typing import Tuple
from src.config import Config
from src.utils.logger import get_logger
from src.ocr.tesseract_ocr import TesseractOCR
from src.ocr.image_preprocessor import ImagePreprocessor
from src.llm.ollama_llm import call_ollama
from src.models.schemas import InvoiceExtraction
from src.utils.date_parser import find_date_in_text

logger = get_logger(__name__)

def try_extract_date_from_header(image_path: Path, ocr: TesseractOCR) -> Tuple[str | None, str]:
    """
    Intenta extraer la fecha directamente de la cabecera del ticket usando un ROI dedicado.

    Returns:
        tuple(fecha_normalizada | None, texto_ocr_cabecera)
    """
    header_img = ImagePreprocessor.extract_header_region(str(image_path))
    if header_img is None:
        return None, ""

    header_text_last = ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        ImagePreprocessor.save_processed_image(header_img, tmp_path)

    header_configs = [
        f"--psm {Config.OCR_PSM} --oem {Config.OCR_OEM}",
        f"--psm 7 --oem {Config.OCR_OEM} -c tessedit_char_whitelist=0123456789:/-",
    ]

    try:
        for cfg in header_configs:
            header_text = ocr.extract_text_with_config(tmp_path, cfg)
            if header_text:
                header_text_last = header_text
                date = find_date_in_text(header_text)
                if date:
                    return date, header_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return None, header_text_last

def main():
    input_dir: Path = Config.INPUT_DIR
    output_dir: Path = Config.OUTPUT_DIR

    # Limpiar directorio de salida si existe
    if output_dir.exists():
        logger.info(f"Limpiando directorio de salida: {output_dir}")
        shutil.rmtree(output_dir)

    # Crear directorio de salida limpio
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida creado: {output_dir}")

    ocr = TesseractOCR(lang=Config.OCR_LANG)

    images = [p for p in sorted(input_dir.iterdir())
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}]

    if not images:
        logger.warning(f"No hay imagenes en {input_dir}")
        return

    for img in images:
        try:
            logger.info(f"OCR: {img.name}")

            # INTENTO 1: OCR básico sin preprocesamiento
            text = ocr.extract_text(str(img))

            # Guarda .txt con el OCR bruto
            txt_path = output_dir / f"{img.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")

            # Llama al LLM para extraccion estructurada
            llm_dict = call_ollama(text)

            # Crea objeto validado
            data = InvoiceExtraction(
                fecha=llm_dict.get("fecha"),
                importe_total=llm_dict.get("importe_total"),
                emisor=llm_dict.get("emisor"),
                concepto=llm_dict.get("concepto")
            )

            # RETRY con preprocesamiento avanzado si falta fecha (dato crítico)
            if data.fecha is None:
                logger.warning(f"⚠ Fecha no encontrada en {img.name}")
                logger.info("Reintentando con preprocesamiento avanzado (SDT + reducción ruido + deskew)...")

                try:
                    # Aplicar preprocesamiento completo
                    text_retry = ocr.extract_with_advanced_preprocessing(str(img))
                    llm_dict_retry = call_ollama(text_retry)

                    if llm_dict_retry.get("fecha"):
                        logger.info(f"  ✓ ÉXITO con preprocesamiento avanzado! Fecha: {llm_dict_retry.get('fecha')}")

                        data.fecha = llm_dict_retry.get("fecha")
                        if llm_dict_retry.get("emisor"):
                            data.emisor = llm_dict_retry.get("emisor")
                        if llm_dict_retry.get("concepto"):
                            data.concepto = llm_dict_retry.get("concepto")
                        if llm_dict_retry.get("importe_total"):
                            data.importe_total = llm_dict_retry.get("importe_total")

                        txt_path.write_text(
                            f"=== OCR con preprocesamiento avanzado ===\n\n{text_retry}",
                            encoding="utf-8"
                        )
                    else:
                        logger.info("  ✗ Preprocesamiento avanzado no encontró fecha, intentando extracción de cabecera...")

                except Exception as e:
                    logger.error(f"  ✗ Error en preprocesamiento avanzado: {e}")

                if data.fecha is None:
                    header_date, header_text = try_extract_date_from_header(img, ocr)
                    if header_date:
                        data.fecha = header_date
                        logger.info(f"✓ Fecha extraída de la cabecera: {data.fecha}")

                        existing_text = txt_path.read_text(encoding="utf-8")
                        updated_text = existing_text + f"\n\n=== OCR cabecera (fecha) ===\n{header_text}"
                        txt_path.write_text(updated_text, encoding="utf-8")
                    else:
                        logger.warning(f"⚠ Todos los intentos fallaron para {img.name}. Fecha: null")
                else:
                    logger.info(f"✓ Fecha recuperada exitosamente: {data.fecha}")

            # Guarda .json estructurado
            json_path = output_dir / f"{img.stem}.json"
            json_path.write_text(
                json.dumps(data.model_dump(), ensure_ascii=False, indent=Config.JSON_INDENT),
                encoding="utf-8"
            )

            logger.info(f"✓ {img.name} -> {txt_path.name}, {json_path.name}")

        except Exception as e:
            logger.exception(f"✗ Error procesando {img.name}: {e}")

if __name__ == "__main__":
    main()
