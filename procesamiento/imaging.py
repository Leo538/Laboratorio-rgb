from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage


ORDEN_CANALES = ("R", "G", "B")
INDICE_CANAL = {"R": 0, "G": 1, "B": 2}
TITULO_CANAL = {"R": "Rojo", "G": "Verde", "B": "Azul"}
COLOR_CANAL = {"R": "#ff6b6b", "G": "#35d07f", "B": "#58a6ff"}


@dataclass(slots=True)
class PerfilMezcla:
    nivel_rojo: int = 100
    nivel_verde: int = 100
    nivel_azul: int = 100


def _recortar_a_byte(valor: float) -> np.uint8:
    if valor < 0:
        return np.uint8(0)
    if valor > 255:
        return np.uint8(255)
    return np.uint8(int(round(valor)))


def _intercambiar_canales_rojo_azul(imagen: np.ndarray) -> np.ndarray:
    if imagen.ndim != 3 or imagen.shape[2] != 3:
        raise ValueError("Se esperaba una imagen de tres canales.")
    return imagen[:, :, [2, 1, 0]].copy()


def abrir_imagen_rgb(ruta: str | Path) -> np.ndarray:
    archivo = Path(ruta)
    buffer = np.fromfile(str(archivo), dtype=np.uint8)
    imagen_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if imagen_bgr is None:
        raise ValueError("No fue posible abrir la imagen seleccionada.")
    return _intercambiar_canales_rojo_azul(imagen_bgr)


def guardar_imagen(ruta: str | Path, imagen: np.ndarray) -> None:
    archivo = Path(ruta)
    extension = archivo.suffix.lower() or ".png"
    if extension not in {".png", ".jpg", ".jpeg", ".bmp"}:
        raise ValueError("Formato de salida no soportado.")

    salida = imagen
    if imagen.ndim == 3:
        salida = _intercambiar_canales_rojo_azul(imagen)

    exito, codificada = cv2.imencode(extension, salida)
    if not exito:
        raise ValueError("No fue posible guardar la imagen.")
    codificada.tofile(str(archivo))


def matriz_a_qimage(imagen: np.ndarray) -> QImage:
    pixeles = np.ascontiguousarray(np.clip(imagen, 0, 255).astype(np.uint8))
    if pixeles.ndim == 2:
        alto, ancho = pixeles.shape
        qimage = QImage(
            pixeles.data,
            ancho,
            alto,
            pixeles.strides[0],
            QImage.Format.Format_Grayscale8,
        )
        return qimage.copy()

    alto, ancho, _ = pixeles.shape
    qimage = QImage(
        pixeles.data,
        ancho,
        alto,
        pixeles.strides[0],
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()


def separar_canales_rgb(imagen: np.ndarray) -> dict[str, np.ndarray]:
    return {
        nombre: imagen[:, :, indice].copy()
        for nombre, indice in INDICE_CANAL.items()
    }


def calcular_histograma(canal: np.ndarray) -> np.ndarray:
    return np.bincount(canal.ravel(), minlength=256).astype(np.float32)


def ajustar_intensidad_canal(canal: np.ndarray, nivel: int) -> np.ndarray:
    factor = max(0, nivel) / 100.0
    ajustado = canal.astype(np.float32) * factor
    return np.clip(ajustado, 0, 255).astype(np.uint8)


def aplicar_perfil_rgb(
    canales: dict[str, np.ndarray],
    perfil: PerfilMezcla,
) -> dict[str, np.ndarray]:
    return {
        "R": ajustar_intensidad_canal(canales["R"], perfil.nivel_rojo),
        "G": ajustar_intensidad_canal(canales["G"], perfil.nivel_verde),
        "B": ajustar_intensidad_canal(canales["B"], perfil.nivel_azul),
    }


def combinar_canales_rgb(canales: dict[str, np.ndarray]) -> np.ndarray:
    imagen = np.zeros((canales["R"].shape[0], canales["R"].shape[1], 3), dtype=np.uint8)
    imagen[:, :, 0] = canales["R"]
    imagen[:, :, 1] = canales["G"]
    imagen[:, :, 2] = canales["B"]
    return imagen


def colorear_canal(canal: np.ndarray, nombre_canal: str) -> np.ndarray:
    alto, ancho = canal.shape
    vista = np.zeros((alto, ancho, 3), dtype=np.uint8)
    vista[:, :, INDICE_CANAL[nombre_canal]] = canal
    return vista


def aplicar_mascara_bloques(
    imagen: np.ndarray,
    mascara_x: int,
    mascara_y: int,
) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    bloques_x = max(1, int(mascara_x))
    bloques_y = max(1, int(mascara_y))
    ancho_reducido = max(1, int(np.ceil(ancho / float(bloques_x))))
    alto_reducido = max(1, int(np.ceil(alto / float(bloques_y))))

    if imagen.ndim == 2:
        reducida = np.zeros((alto_reducido, ancho_reducido), dtype=np.uint8)
    else:
        reducida = np.zeros((alto_reducido, ancho_reducido, imagen.shape[2]), dtype=np.uint8)

    for y_reducido in range(alto_reducido):
        y0 = int(y_reducido * alto / alto_reducido)
        y1 = int((y_reducido + 1) * alto / alto_reducido)
        y1 = max(y0 + 1, min(alto, y1))

        for x_reducido in range(ancho_reducido):
            x0 = int(x_reducido * ancho / ancho_reducido)
            x1 = int((x_reducido + 1) * ancho / ancho_reducido)
            x1 = max(x0 + 1, min(ancho, x1))
            bloque = imagen[y0:y1, x0:x1]

            if imagen.ndim == 2:
                suma = 0
                conteo = 0
                for y in range(bloque.shape[0]):
                    for x in range(bloque.shape[1]):
                        suma += int(bloque[y, x])
                        conteo += 1
                reducida[y_reducido, x_reducido] = _recortar_a_byte(suma / max(1, conteo))
                continue

            suma_r = 0
            suma_g = 0
            suma_b = 0
            conteo = 0
            for y in range(bloque.shape[0]):
                for x in range(bloque.shape[1]):
                    suma_r += int(bloque[y, x, 0])
                    suma_g += int(bloque[y, x, 1])
                    suma_b += int(bloque[y, x, 2])
                    conteo += 1

            reducida[y_reducido, x_reducido, 0] = _recortar_a_byte(suma_r / max(1, conteo))
            reducida[y_reducido, x_reducido, 1] = _recortar_a_byte(suma_g / max(1, conteo))
            reducida[y_reducido, x_reducido, 2] = _recortar_a_byte(suma_b / max(1, conteo))

    salida = np.zeros_like(imagen)
    for y in range(alto):
        y_reducido = min(alto_reducido - 1, int(y * alto_reducido / alto))
        for x in range(ancho):
            x_reducido = min(ancho_reducido - 1, int(x * ancho_reducido / ancho))
            salida[y, x] = reducida[y_reducido, x_reducido]

    return salida


def convertir_a_grises(imagen_rgb: np.ndarray) -> np.ndarray:
    r = imagen_rgb[:, :, 0].astype(np.float32)
    g = imagen_rgb[:, :, 1].astype(np.float32)
    b = imagen_rgb[:, :, 2].astype(np.float32)
    gris = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return np.clip(gris, 0, 255).astype(np.uint8)


def binarizar_imagen(imagen_gris: np.ndarray, umbral: int) -> np.ndarray:
    umbral_ajustado = int(np.clip(umbral, 0, 255))
    return np.where(imagen_gris >= umbral_ajustado, 255, 0).astype(np.uint8)


def describir_canal(canal: np.ndarray) -> str:
    return (
        f"min={int(canal.min())} | max={int(canal.max())} | "
        f"media={float(canal.mean()):.1f}"
    )


def imagen_es_gris(imagen_rgb: np.ndarray, tolerancia: int = 2) -> bool:
    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] != 3:
        return True
    imagen_int = imagen_rgb.astype(np.int16)
    diferencia = imagen_int.max(axis=2) - imagen_int.min(axis=2)
    return int(diferencia.max()) <= tolerancia
