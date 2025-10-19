"""
Sistema completo de preprocesamiento para OCR basado en algoritmos avanzados.
Incluye detección de ruido SDT, reducción de ruido gaussiano/impulso,
corrección de orientación, eliminación de marcas de agua y pipeline integrado.
"""

import cv2
import numpy as np
import scipy.signal
import math
import statistics
from typing import Tuple, Optional


class NoiseTypeDetector:
    """
    Detecta automáticamente el tipo de ruido en una imagen usando el algoritmo SDT
    (Spike Detection Test).
    """

    # Umbrales calibrados empíricamente
    LOWER_GAUSSIAN = 96.04917198684099
    UPPER_GAUSSIAN = 326.5743861507359
    LOWER_IMPULSE = 4039.43981828374
    UPPER_IMPULSE = 8989.931753143906

    def __init__(self, image: np.ndarray):
        """
        Args:
            image: Imagen en escala de grises (numpy array)
        """
        self.image = image

    def _calculate_sdt_distance(self) -> float:
        """
        Calcula la distancia SDT basada en el histograma de la imagen.

        Returns:
            Distancia SDT normalizada
        """
        # PASO 1: Calcular histograma
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        H = hist.flatten().tolist()

        # PASO 2: Calcular diferencias entre bins consecutivos
        D = [H[i + 1] - H[i] for i in range(len(H) - 1)]

        # PASO 3: Calcular distancia SDT
        NL1 = min(D)
        NL2 = max(D)
        distance = NL2 - NL1

        # Normalizar por el número de píxeles
        height, width = self.image.shape
        distance_normalized = (distance / (width * height)) * 100000

        return distance_normalized

    def flag(self) -> Tuple[int, int]:
        """
        Clasifica el tipo de ruido presente en la imagen.

        Returns:
            Tupla (gaussian_flag, impulse_flag) donde:
            - (1, 0): Ruido gaussiano detectado
            - (0, 1): Ruido de impulso detectado
            - (0, 0): Sin ruido significativo
        """
        distance = self._calculate_sdt_distance()

        # PASO 4: Clasificar según umbrales
        if self.LOWER_GAUSSIAN <= distance <= self.UPPER_GAUSSIAN:
            return (1, 0)  # Ruido gaussiano
        elif self.LOWER_IMPULSE <= distance <= self.UPPER_IMPULSE:
            return (0, 1)  # Ruido de impulso
        else:
            return (0, 0)  # Sin ruido


class NoiseReductionApply:
    """
    Aplica diferentes técnicas de reducción de ruido según el tipo detectado.
    """

    def __init__(self, image: np.ndarray):
        """
        Args:
            image: Imagen en escala de grises (numpy array)
        """
        self.image = image.copy()

    def gaussian_blur(self) -> np.ndarray:
        """
        Aplica Gaussian Blur simple con OpenCV.

        Returns:
            Imagen con ruido gaussiano reducido
        """
        return cv2.GaussianBlur(self.image, (5, 5), 0)

    def _get_gaussian_noise_sd(self) -> float:
        """
        Calcula la desviación estándar del ruido gaussiano mediante convolución.

        Returns:
            Desviación estándar estimada del ruido
        """
        M, N = self.image.shape
        pi_by_2_sqrt = math.sqrt(math.pi / 2)

        # Máscara para detección de ruido
        MASK = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], dtype=np.float64)

        # Convolucionar imagen con la máscara
        convolved = scipy.signal.convolve2d(
            self.image.astype(np.float64),
            MASK,
            mode='same',
            boundary='fill',
            fillvalue=0
        )

        # Calcular desviación estándar del ruido
        gaussian_noise_sd = pi_by_2_sqrt * (1 / (6 * M * N)) * np.sum(np.abs(convolved))

        return gaussian_noise_sd

    def paper_algo_gaussian_removal(self, smoothing_factor: int = 5, W: int = 3) -> np.ndarray:
        """
        Implementa el algoritmo del paper para reducción de ruido gaussiano.
        Filtra píxel por píxel basándose en vecinos similares.

        Args:
            smoothing_factor: Factor de suavizado (mayor = más suavizado)
            W: Tamaño de ventana (siempre 3 para ventana 3x3)

        Returns:
            Imagen con ruido gaussiano reducido
        """
        # PASO 1: Calcular desviación estándar del ruido
        gaussian_noise_sd = self._get_gaussian_noise_sd()

        # PASO 2: Aplicar filtro adaptativo píxel por píxel
        img = self.image.copy().astype(np.float64)
        threshold = (2 * W) - 1  # = 5

        rows, cols = img.shape

        # Procesar cada píxel (excluyendo bordes)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # a) Extraer ventana 3x3
                image_segment = [
                    img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                    img[i, j - 1], img[i, j], img[i, j + 1],
                    img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]
                ]

                # b) Píxel central
                centre_pixel = image_segment[4]

                # c) Calcular diferencias absolutas con vecinos
                neighbors_indices = [0, 1, 2, 3, 5, 6, 7, 8]  # Excluir centro (índice 4)
                absolute_differences = [
                    abs(image_segment[idx] - centre_pixel)
                    for idx in neighbors_indices
                ]

                # d) Filtrar vecinos similares
                DA = []
                count = 0
                threshold_diff = gaussian_noise_sd * smoothing_factor

                for idx, diff in zip(neighbors_indices, absolute_differences):
                    if diff < threshold_diff:
                        DA.append(image_segment[idx])
                        count += 1

                # e) Decidir si reemplazar el píxel
                if count > 0:
                    mean_of_DA = statistics.mean(DA)
                    if count > threshold:
                        img[i, j] = mean_of_DA

        return img.astype(np.uint8)

    def median_blur(self) -> np.ndarray:
        """
        Aplica Median Blur para reducir ruido de impulso (salt & pepper).

        Returns:
            Imagen con ruido de impulso reducido
        """
        return cv2.medianBlur(self.image, 3)

    def thresholding(self) -> np.ndarray:
        """
        Aplica binarización con método de Otsu.

        Returns:
            Imagen binarizada
        """
        _, binary = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


class OrientationCorrection:
    """
    Corrige la orientación (deskew) de documentos inclinados.
    """

    def __init__(self, image: np.ndarray):
        """
        Args:
            image: Imagen en escala de grises (numpy array)
        """
        self.image = image

    def _get_skewed_angle(self) -> float:
        """
        Detecta el ángulo de inclinación del documento.

        Returns:
            Ángulo de inclinación en grados
        """
        # PASO 1: Preprocesamiento
        blur = cv2.GaussianBlur(self.image, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # PASO 2: Morfología para detectar líneas de texto
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        # PASO 3: Encontrar contornos
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Ordenar por área y tomar el mayor
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # PASO 4: Calcular rectángulo mínimo de área
        min_area_rect = cv2.minAreaRect(largest_contour)
        angle = min_area_rect[2]

        # PASO 5: Normalizar ángulo
        if angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        skew_angle = round(-1.0 * angle, 2)

        return skew_angle

    def orientation_correction(self) -> np.ndarray:
        """
        Corrige la orientación de la imagen rotándola según el ángulo detectado.

        Returns:
            Imagen con orientación corregida
        """
        # PASO 1: Obtener dimensiones y centro
        rows, cols = self.image.shape
        img_center = (cols / 2, rows / 2)

        # PASO 2: Calcular ángulo de inclinación
        skew_angle = self._get_skewed_angle()

        # PASO 3: Crear matriz de rotación
        M = cv2.getRotationMatrix2D(img_center, skew_angle, 1)

        # PASO 4: Aplicar rotación con fondo blanco
        rotated_image = cv2.warpAffine(
            self.image,
            M,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated_image


class WatermarkRemoval:
    """
    Elimina marcas de agua de baja frecuencia espacial de documentos.
    """

    def __init__(self, image_path: str):
        """
        Args:
            image_path: Ruta al archivo de imagen
        """
        self.input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.input_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    def output(self) -> np.ndarray:
        """
        Elimina la marca de agua mediante filtrado y sustracción.

        Returns:
            Imagen sin marca de agua
        """
        # PASO 1: Crear versión difuminada (preserva marca de agua)
        watermark = cv2.medianBlur(self.input_image, 19)

        # PASO 2: Restar original de difuminada (elimina marca de agua)
        output = cv2.subtract(watermark, self.input_image)

        # PASO 3: Invertir bits (restaurar polaridad)
        output = cv2.bitwise_not(output)

        return output


class ImagePreprocessor:
    """
    Pipeline completo de preprocesamiento para OCR.
    Integra todos los módulos de procesamiento de imagen.
    """

    @staticmethod
    def preprocess_for_ocr(
        image_path: str,
        remove_watermark: bool = False,
        apply_deskew: bool = True,
        apply_final_threshold: bool = True
    ) -> np.ndarray:
        """
        Aplica el pipeline completo de preprocesamiento.

        Args:
            image_path: Ruta a la imagen de entrada
            remove_watermark: Si se debe intentar eliminar marcas de agua
            apply_deskew: Si se debe corregir la orientación
            apply_final_threshold: Si se debe aplicar binarización final

        Returns:
            Imagen preprocesada lista para OCR
        """
        # PASO 1: Cargar imagen en escala de grises
        if remove_watermark:
            # Si vamos a eliminar marca de agua, usar esa clase
            watermark_remover = WatermarkRemoval(image_path)
            img = watermark_remover.output()
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # PASO 2: Detección de tipo de ruido
        detector = NoiseTypeDetector(img)
        gaussian_flag, impulse_flag = detector.flag()

        # PASO 3: Reducción de ruido (condicional)
        noise_reducer = NoiseReductionApply(img)

        if gaussian_flag == 1:
            # Ruido gaussiano detectado - usar algoritmo del paper
            img = noise_reducer.paper_algo_gaussian_removal()

        if impulse_flag == 1:
            # Ruido de impulso detectado - usar median blur
            img = cv2.medianBlur(img, 3)

        # PASO 4: Corrección de orientación
        if apply_deskew:
            orientation_corrector = OrientationCorrection(img)
            img = orientation_corrector.orientation_correction()

        # PASO 5: Binarización final (opcional pero recomendado)
        if apply_final_threshold:
            noise_reducer_final = NoiseReductionApply(img)
            img = noise_reducer_final.thresholding()

        return img

    @staticmethod
    def save_processed_image(img: np.ndarray, output_path: str):
        """Guarda imagen procesada en disco."""
        cv2.imwrite(output_path, img)

    @staticmethod
    def extract_header_region(image_path: str, top_percentage: float = 0.35) -> Optional[np.ndarray]:
        """
        Extrae una región superior del documento preprocesado.
        Útil para buscar fechas en la cabecera del documento.

        Args:
            image_path: Ruta a la imagen
            top_percentage: Porcentaje superior a extraer (0.0 a 1.0)

        Returns:
            Región superior de la imagen preprocesada o None si falla
        """
        try:
            img = ImagePreprocessor.preprocess_for_ocr(image_path)
            if img.size == 0:
                return None

            # Extraer región superior
            header_height = max(int(img.shape[0] * top_percentage), 1)
            header = img[:header_height, :]

            return header
        except Exception:
            return None
