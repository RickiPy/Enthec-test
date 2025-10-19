# Document Processor - Sistema de Extraccion de Facturas con OCR y LLM

Sistema automatizado para procesamiento de imagenes de facturas y tickets que combina OCR (Tesseract) con preprocesamiento avanzado de imagenes y extraccion estructurada mediante LLM (Ollama).

## Tabla de Contenidos

- [Caracteristicas](#caracteristicas)
- [Requisitos Previos](#requisitos-previos)
- [Instalacion](#instalacion)
- [Configuracion](#configuracion)
- [Uso](#uso)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Pipeline de Procesamiento](#pipeline-de-procesamiento)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Testing](#testing)

## Caracteristicas

- **OCR Avanzado**: Tesseract OCR con preprocesamiento inteligente de imagenes
- **Deteccion Automatica de Ruido**: Algoritmo SDT (Spike Detection Test) para identificar ruido gaussiano e impulso
- **Correccion de Orientacion**: Deskew automatico de documentos inclinados
- **Extraccion Estructurada**: LLM (Ollama) para extraer campos especificos (fecha, importe, emisor, concepto)
- **Retry Inteligente**: Sistema de reintentos progresivos cuando la extraccion falla
- **Validacion con Pydantic**: Esquemas validados para datos estructurados

## Requisitos Previos

### 1. Tesseract OCR

Tesseract es el motor OCR de codigo abierto que reconoce texto en las imagenes.

#### Instalacion:

**Windows:**
1. Descargar instalador desde: https://github.com/UB-Mannheim/tesseract/wiki
2. Ejecutar el instalador
3. Anadir al PATH o configurar la ruta en `src/config.py`
4. Ubicacion tipica: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng  # English language pack (usually included by default)
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # Para idiomas adicionales
```

**Verificar instalacion:**
```bash
tesseract --version
```

**Sitio oficial:** https://github.com/tesseract-ocr/tesseract

**Nota:** Este proyecto fue desarrollado y probado con Python 3.12.5

### 2. Ollama (LLM Local)

Ollama permite ejecutar modelos de lenguaje localmente para la extraccion estructurada.

#### Instalacion:

**Windows/macOS/Linux:**
1. Descargar desde: https://ollama.com/download
2. Instalar siguiendo las instrucciones del sistema operativo
3. Descargar el modelo requerido:

```bash
ollama pull llama3.2:3b
```

**Verificar instalacion:**
```bash
ollama --version
ollama list  # Debe mostrar llama3.2:3b
```

**Sitio oficial:** https://ollama.com

### 3. Docker y Docker Compose (Opcional - para servicios adicionales)

Si el proyecto requiere servicios adicionales en contenedores:

```bash
docker-compose up -d
```

Los servicios estaran disponibles en:
- **Ollama API**: http://localhost:11434 (puerto por defecto de Ollama)

**Nota:** Ollama puede ejecutarse directamente en el host sin Docker, usando su instalacion nativa.

## Instalacion

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd document-processor
```

### 2. Crear entorno virtual

**Requisitos:** Python 3.12.5 o superior

```bash
python -m venv venv

# Activar el entorno:
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `pytesseract` - Wrapper Python para Tesseract OCR
- `opencv-python` - Procesamiento de imagenes
- `numpy` - Operaciones numericas
- `scipy` - Algoritmos cientificos (convolucion para deteccion de ruido)
- `pillow` - Manipulacion de imagenes
- `ollama` - Cliente Python para Ollama
- `pydantic>=2` - Validacion de datos
- `python-dateutil` - Parsing de fechas
- `pytest` - Testing

## Configuracion

### 1. Configurar rutas de Tesseract (Windows)

Editar `src/config.py`:

```python
class Config:
    # Ruta de Tesseract (solo necesario en Windows si no esta en PATH)
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Directorios de entrada/salida
    INPUT_DIR = Path("data/test-images")
    OUTPUT_DIR = Path("data/outputs")

    # Configuracion OCR
    OCR_LANG = "eng"  # Language: 'eng' for English (default)
    OCR_PSM = 6       # Page Segmentation Mode
    OCR_OEM = 3       # OCR Engine Mode

    # Configuracion LLM
    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_BASE_URL = "http://localhost:11434"
```

### 2. Preparar directorios

```bash
mkdir -p data/test-images
mkdir -p data/outputs
```

### 3. Colocar imagenes a procesar

Colocar archivos de imagenes (JPG, PNG, TIF, etc.) en `data/test-images/`

## Uso

### Ejecucion basica

```bash
python -m src.main
```

El sistema procesara todas las imagenes en `data/test-images/` y generara:
- **Archivos `.txt`**: Texto extraido por OCR
- **Archivos `.json`**: Datos estructurados (fecha, importe, emisor, concepto)

**Nota importante:** Cada ejecucion limpia automaticamente el directorio `data/outputs/` para evitar mezclar resultados de diferentes ejecuciones.

### Ejemplo de salida JSON

```json
{
    "fecha": "2024-10-15",
    "importe_total": "45.50",
    "emisor": "Supermercado XYZ",
    "concepto": "Compra de alimentos"
}
```

## Arquitectura del Sistema

### Modulos Principales

```
src/
├── ocr/
│   ├── image_preprocessor.py    # Sistema de preprocesamiento de imagenes
│   └── tesseract_ocr.py          # Motor OCR con integracion de preprocesamiento
├── llm/
│   └── ollama_llm.py             # Cliente LLM para extraccion estructurada
├── models/
│   └── schemas.py                # Esquemas Pydantic de validacion
├── utils/
│   ├── logger.py                 # Sistema de logging
│   └── date_parser.py            # Parser de fechas
├── config.py                     # Configuracion centralizada
└── main.py                       # Pipeline principal
```

## Pipeline de Procesamiento

### Flujo de Trabajo

```
1. Carga de imagen
   |
2. OCR basico (Tesseract directo)
   |
3. Extraccion LLM (intento inicial)
   |
4. Fecha extraida? --> SI --> Guardar resultados
   |
   NO
   |
5. Preprocesamiento avanzado:
   |
   |--> Deteccion de ruido (algoritmo SDT)
   |    - Calcula histograma
   |    - Clasifica: gaussiano / impulso / sin ruido
   |
   |--> Reduccion de ruido
   |    - Gaussiano: Algoritmo adaptativo pixel por pixel
   |    - Impulso: Median Blur
   |
   |--> Correccion de orientacion (Deskew)
   |    - Deteccion de angulo mediante contornos
   |    - Rotacion con interpolacion
   |
   |--> Binarizacion (Otsu)
   |
6. OCR en imagen preprocesada
   |
7. Extraccion LLM (segundo intento)
   |
8. Fecha extraida? --> SI --> Guardar resultados
   |
   NO
   |
9. Extraccion de cabecera (ROI superior 35%)
   |
10. OCR con configuraciones especiales
    |
11. Busqueda de fecha en cabecera
    |
12. Guardar resultados finales
```

### Sistema de Preprocesamiento de Imagenes

El modulo `image_preprocessor.py` implementa tecnicas avanzadas de procesamiento de imagenes basadas en el repositorio [OCR-with-Tesseract](https://github.com/Purefekt/OCR-with-Tesseract).

#### Componentes:

**1. NoiseTypeDetector**
- Algoritmo SDT (Spike Detection Test)
- Detecta automaticamente ruido gaussiano e impulso
- Umbrales calibrados empiricamente

**2. NoiseReductionApply**
- **Ruido Gaussiano**: Algoritmo adaptativo con convolucion y filtrado por vecindad
- **Ruido Impulso**: Median Blur (kernel 3x3)
- **Binarizacion**: Metodo de Otsu

**3. OrientationCorrection**
- Deteccion de angulo mediante morfologia y contornos
- Rotacion automatica con preservacion de contenido
- Fondo blanco en regiones vacias

**4. WatermarkRemoval**
- Filtrado por frecuencia espacial
- Eliminacion de marcas de agua de baja frecuencia

**5. ImagePreprocessor (Pipeline Integrado)**
```python
from src.ocr.image_preprocessor import ImagePreprocessor

# Pipeline completo automatico
img = ImagePreprocessor.preprocess_for_ocr('imagen.jpg')

# Con opciones personalizadas
img = ImagePreprocessor.preprocess_for_ocr(
    'imagen.jpg',
    remove_watermark=True,
    apply_deskew=True,
    apply_final_threshold=True
)
```

**Creditos del preprocesamiento:** Este modulo esta basado en el excelente trabajo del repositorio [Purefekt/OCR-with-Tesseract](https://github.com/Purefekt/OCR-with-Tesseract), adaptado e integrado para este sistema.

## Estructura del Proyecto

```
document-processor/
├── data/
│   ├── test-images/          # Imagenes de entrada
│   └── outputs/              # Resultados (.txt y .json)
├── src/
│   ├── ocr/
│   │   ├── image_preprocessor.py
│   │   └── tesseract_ocr.py
│   ├── llm/
│   │   └── ollama_llm.py
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── date_parser.py
│   ├── config.py
│   └── main.py
├── tests/
│   ├── test_ocr.py
│   ├── test_llm.py
│   ├── test_models.py
│   └── test_utils.py
├── requirements.txt
├── README.md
├── PREPROCESSING_GUIDE.md    # Guia detallada del preprocesamiento
└── docker-compose.yml        # (Opcional) Servicios Docker
```

## Testing

### Ejecutar todos los tests

```bash
pytest
```

### Ejecutar tests especificos

```bash
# Tests de OCR
pytest tests/test_ocr.py -v

# Tests de LLM
pytest tests/test_llm.py -v

# Tests de modelos
pytest tests/test_models.py -v

# Tests de utilidades
pytest tests/test_utils.py -v
```

### Tests implementados

- **OCR**: 20 tests cubriendo Tesseract y preprocesamiento
- **LLM**: 4 tests de integracion con Ollama
- **Schemas**: 15 tests de validacion de Pydantic
- **Utils**: 5 tests de logger y parser de fechas

**Total: 44 tests unitarios**

## Ejemplos de Uso Avanzado

### Procesar una sola imagen con logging detallado

```python
from src.ocr.tesseract_ocr import TesseractOCR
from src.ocr.image_preprocessor import ImagePreprocessor
from src.llm.ollama_llm import call_ollama

# 1. Preprocesar imagen
img = ImagePreprocessor.preprocess_for_ocr('factura.jpg')
ImagePreprocessor.save_processed_image(img, 'factura_procesada.png')

# 2. Ejecutar OCR
ocr = TesseractOCR(lang='eng')
text = ocr.extract_text('factura_procesada.png')

print("Texto extraido:")
print(text)

# 3. Extraer datos estructurados
data = call_ollama(text)
print("\nDatos estructurados:")
print(data)
```

### Usar deteccion de ruido manual

```python
import cv2
from src.ocr.image_preprocessor import NoiseTypeDetector, NoiseReductionApply

# Cargar imagen
img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Detectar tipo de ruido
detector = NoiseTypeDetector(img)
gaussian_flag, impulse_flag = detector.flag()

print(f"Ruido gaussiano: {gaussian_flag}")
print(f"Ruido de impulso: {impulse_flag}")

# Aplicar reduccion especifica
reducer = NoiseReductionApply(img)
if gaussian_flag:
    img_clean = reducer.paper_algo_gaussian_removal()
elif impulse_flag:
    img_clean = reducer.median_blur()
```

## Troubleshooting

### Error: "Tesseract not found"
**Solucion:** Configurar `TESSERACT_CMD` en `src/config.py` con la ruta completa de Tesseract.

### Error: "Ollama connection refused"
**Solucion:**
1. Verificar que Ollama esta ejecutandose: `ollama serve`
2. Verificar puerto en `src/config.py`: `OLLAMA_BASE_URL = "http://localhost:11434"`

### OCR no detecta texto
**Solucion:**
1. Verificar que la imagen tenga suficiente contraste
2. Probar con `apply_final_threshold=False` en el preprocesamiento
3. Verificar el idioma configurado en `OCR_LANG`

### El preprocesamiento es muy lento
**Solucion:** El algoritmo de reduccion de ruido gaussiano procesa pixel por pixel. Para imagenes grandes, considerar desactivarlo o usar `gaussian_blur()` simple.


---

**Desarrollado para demostrar integracion OCR + LLM + Preprocesamiento Avanzado**
