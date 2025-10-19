import pytesseract
import cv2
import tempfile
import os
from PIL import Image, ImageOps
from src.config import Config
from src.utils.logger import get_logger
from src.ocr.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = getattr(
    Config, "TESSERACT_CMD", pytesseract.pytesseract.tesseract_cmd
)


class TesseractOCR:
    def __init__(self, lang="eng"):
        """Inicializa el motor OCR con idioma por defecto."""
        self.lang = lang
        self._default_config = f"--psm {Config.OCR_PSM} --oem {Config.OCR_OEM}"

    @staticmethod
    def _prepare_image(image_path: str) -> Image.Image:
        """Carga la imagen respetando EXIF y convierte a grises."""
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        return img

    def extract_text(self, image_path: str) -> str:
        """Extrae texto de una imagen usando Tesseract OCR (método básico)."""
        try:
            img = self._prepare_image(image_path)
            text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=self._default_config,
            )
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error al procesar la imagen {image_path}: {e}")

    def extract_text_with_config(self, image_path: str, config: str) -> str:
        """Ejecuta OCR con una configuración personalizada."""
        try:
            img = self._prepare_image(image_path)
            text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=config,
            )
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error al procesar la imagen {image_path} con config personalizada: {e}")

    def extract_with_retry(self, image_path: str) -> tuple[str, bool]:
        """
        Intenta OCR básico primero. Si falla o da poco texto, usa preprocesamiento avanzado.

        Returns:
            tuple: (texto, usó_preprocesamiento)
        """
        try:
            # Intento 1: OCR básico
            text_basic = self.extract_text(image_path)

            # Si el texto es muy corto, probablemente falló el OCR
            if len(text_basic) < 50:
                logger.warning(f"OCR básico dio poco texto ({len(text_basic)} chars), reintentando con preprocesamiento avanzado...")
                text_enhanced = self.extract_with_advanced_preprocessing(image_path)
                return text_enhanced, True
            else:
                return text_basic, False

        except Exception as e:
            logger.error(f"Error en extract_with_retry: {e}")
            raise RuntimeError(f"Error al procesar la imagen {image_path}: {e}")

    def extract_with_advanced_preprocessing(self, image_path: str) -> str:
        """
        Extrae texto usando el pipeline completo de preprocesamiento avanzado:
        - Detección automática de ruido (SDT)
        - Reducción de ruido gaussiano/impulso
        - Corrección de orientación (deskew)
        - Binarización con Otsu

        Returns:
            Texto extraído de la imagen preprocesada
        """
        try:
            # Aplicar pipeline completo de preprocesamiento
            preprocessed = ImagePreprocessor.preprocess_for_ocr(
                image_path,
                remove_watermark=False,
                apply_deskew=True,
                apply_final_threshold=True
            )

            # Guardar imagen preprocesada temporalmente
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, preprocessed)

            try:
                # Ejecutar OCR en la imagen preprocesada
                text = pytesseract.image_to_string(
                    tmp_path,
                    lang=self.lang,
                    config=self._default_config
                )
                return text.strip()
            finally:
                # Limpiar archivo temporal
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error(f"Error en preprocesamiento avanzado: {e}")
            raise RuntimeError(f"Error al preprocesar imagen {image_path}: {e}")
