from pathlib import Path
import os

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "test-images"
    OUTPUT_DIR = DATA_DIR / "outputs"

    # OCR
    OCR_LANG = os.getenv("OCR_LANG", "eng")  # Cambiar a "spa" cuando tengas los datos de español
    OCR_PSM = int(os.getenv("OCR_PSM", "6"))
    OCR_OEM = int(os.getenv("OCR_OEM", "3"))
    # Ruta de Tesseract en Windows
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # LLM (Ollama en Docker expuesto en localhost:11434)
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Salida
    JSON_INDENT = 2
