from pathlib import Path
from src.ocr.tesseract_ocr import TesseractOCR
from src.ocr.image_preprocessor import (
    NoiseTypeDetector,
    NoiseReductionApply,
    OrientationCorrection,
    ImagePreprocessor
)
import pytest
from PIL import Image
import numpy as np
import cv2

def test_ocr_file_not_found(tmp_path: Path):
    """Test que lanza RuntimeError cuando el archivo no existe"""
    ocr = TesseractOCR(lang="eng")
    with pytest.raises(RuntimeError):
        ocr.extract_text(str(tmp_path / "nope.jpg"))

def test_ocr_initialization():
    """Test inicialización del OCR con diferentes idiomas"""
    ocr_eng = TesseractOCR(lang="eng")
    assert ocr_eng.lang == "eng"

    ocr_spa = TesseractOCR(lang="spa")
    assert ocr_spa.lang == "spa"

def test_ocr_initialization_default_language():
    """Test que el idioma por defecto es ingles"""
    ocr = TesseractOCR()
    assert ocr.lang == "eng"

def test_ocr_with_simple_image(tmp_path: Path):
    """Test OCR con una imagen simple generada"""
    # Crea una imagen de prueba simple (puede no tener texto reconocible)
    img = Image.new('RGB', (100, 100), color='white')
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    ocr = TesseractOCR(lang="eng")

    # Solo verifica que no lance error (el resultado puede ser vacío)
    try:
        result = ocr.extract_text(str(img_path))
        assert isinstance(result, str)  # Debe devolver un string
    except RuntimeError as e:
        # Si Tesseract no está instalado, el test pasará
        pytest.skip(f"Tesseract no disponible: {e}")

def test_ocr_invalid_image_path():
    """Test con ruta de imagen inválida"""
    ocr = TesseractOCR(lang="eng")
    with pytest.raises(RuntimeError):
        ocr.extract_text("/ruta/invalida/imagen.jpg")

def test_ocr_empty_path():
    """Test con ruta vacía"""
    ocr = TesseractOCR(lang="eng")
    with pytest.raises(RuntimeError):
        ocr.extract_text("")

# ===== TESTS PARA NUEVO SISTEMA DE PREPROCESAMIENTO =====

def test_noise_detector_initialization():
    """Test inicialización del detector de ruido"""
    img = np.zeros((100, 100), dtype=np.uint8)
    detector = NoiseTypeDetector(img)
    assert detector.image is not None
    assert detector.image.shape == (100, 100)

def test_noise_detector_flag_returns_tuple():
    """Test que el detector devuelve tupla (gaussian, impulse)"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    detector = NoiseTypeDetector(img)
    gaussian_flag, impulse_flag = detector.flag()

    assert isinstance(gaussian_flag, int)
    assert isinstance(impulse_flag, int)
    assert gaussian_flag in [0, 1]
    assert impulse_flag in [0, 1]

def test_noise_reduction_initialization():
    """Test inicialización del reductor de ruido"""
    img = np.zeros((100, 100), dtype=np.uint8)
    reducer = NoiseReductionApply(img)
    assert reducer.image is not None

def test_noise_reduction_gaussian_blur():
    """Test Gaussian blur simple"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    reducer = NoiseReductionApply(img)
    result = reducer.gaussian_blur()

    assert result is not None
    assert result.shape == img.shape
    assert isinstance(result, np.ndarray)

def test_noise_reduction_median_blur():
    """Test Median blur para ruido de impulso"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    reducer = NoiseReductionApply(img)
    result = reducer.median_blur()

    assert result is not None
    assert result.shape == img.shape
    assert isinstance(result, np.ndarray)

def test_noise_reduction_thresholding():
    """Test binarización con Otsu"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    reducer = NoiseReductionApply(img)
    result = reducer.thresholding()

    assert result is not None
    assert result.shape == img.shape
    # Verificar que es binaria (solo 0 y 255)
    unique_values = np.unique(result)
    assert len(unique_values) <= 2

def test_orientation_correction_initialization():
    """Test inicialización de corrección de orientación"""
    img = np.zeros((100, 100), dtype=np.uint8)
    corrector = OrientationCorrection(img)
    assert corrector.image is not None

def test_orientation_correction_returns_image():
    """Test que la corrección de orientación devuelve imagen"""
    img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    corrector = OrientationCorrection(img)
    result = corrector.orientation_correction()

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape

def test_image_preprocessor_with_temp_image(tmp_path: Path):
    """Test pipeline completo de preprocesamiento"""
    # Crear imagen de prueba
    img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    # Aplicar preprocesamiento
    result = ImagePreprocessor.preprocess_for_ocr(str(img_path))

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 2  # Debe ser escala de grises

def test_image_preprocessor_save_image(tmp_path: Path):
    """Test guardar imagen procesada"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    output_path = tmp_path / "output.png"

    ImagePreprocessor.save_processed_image(img, str(output_path))

    assert output_path.exists()
    loaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    assert loaded.shape == img.shape

def test_image_preprocessor_extract_header(tmp_path: Path):
    """Test extracción de región de cabecera"""
    # Crear imagen de prueba
    img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    # Extraer cabecera
    header = ImagePreprocessor.extract_header_region(str(img_path), top_percentage=0.35)

    assert header is not None
    assert isinstance(header, np.ndarray)
    # Verificar que la altura es aproximadamente 35% de la original
    assert header.shape[0] <= int(200 * 0.35) + 1

def test_image_preprocessor_with_options(tmp_path: Path):
    """Test pipeline con diferentes opciones"""
    img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    # Sin deskew
    result1 = ImagePreprocessor.preprocess_for_ocr(str(img_path), apply_deskew=False)
    assert result1 is not None

    # Sin binarización final
    result2 = ImagePreprocessor.preprocess_for_ocr(str(img_path), apply_final_threshold=False)
    assert result2 is not None

def test_ocr_with_advanced_preprocessing(tmp_path: Path):
    """Test OCR con preprocesamiento avanzado"""
    # Crear imagen simple con texto
    img = Image.new('RGB', (200, 100), color='white')
    img_path = tmp_path / "test_text.png"
    img.save(img_path)

    ocr = TesseractOCR(lang="eng")

    try:
        result = ocr.extract_with_advanced_preprocessing(str(img_path))
        assert isinstance(result, str)
    except RuntimeError as e:
        pytest.skip(f"Tesseract no disponible: {e}")
