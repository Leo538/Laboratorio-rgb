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


def abrir_imagen_rgb(ruta: str | Path) -> np.ndarray:
    archivo = Path(ruta)
    buffer = np.fromfile(str(archivo), dtype=np.uint8)
    imagen_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if imagen_bgr is None:
        raise ValueError("No fue posible abrir la imagen seleccionada.")
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)


def guardar_imagen(ruta: str | Path, imagen: np.ndarray) -> None:
    archivo = Path(ruta)
    extension = archivo.suffix.lower() or ".png"
    if extension not in {".png", ".jpg", ".jpeg", ".bmp"}:
        raise ValueError("Formato de salida no soportado.")

    salida = imagen
    if imagen.ndim == 3:
        salida = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

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
    histograma = cv2.calcHist([canal], [0], None, [256], [0, 256])
    return histograma.reshape(-1).astype(np.float32)


def ajustar_intensidad_canal(canal: np.ndarray, nivel: int) -> np.ndarray:
    factor = max(0, nivel) / 100.0
    return cv2.convertScaleAbs(canal, alpha=factor, beta=0)


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
    return cv2.merge([canales["R"], canales["G"], canales["B"]])


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

    reducida = cv2.resize(
        imagen,
        (ancho_reducido, alto_reducido),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(
        reducida,
        (ancho, alto),
        interpolation=cv2.INTER_NEAREST,
    )


def convertir_a_grises(imagen_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)


def binarizar_imagen(imagen_gris: np.ndarray, umbral: int) -> np.ndarray:
    _, binaria = cv2.threshold(
        imagen_gris,
        int(np.clip(umbral, 0, 255)),
        255,
        cv2.THRESH_BINARY,
    )
    return binaria


def describir_canal(canal: np.ndarray) -> str:
    return (
        f"min={int(canal.min())} | max={int(canal.max())} | "
        f"media={float(canal.mean()):.1f}"
    )


def imagen_es_gris(imagen_rgb: np.ndarray, tolerancia: int = 2) -> bool:
    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] != 3:
        return True
    dif_rg = cv2.absdiff(imagen_rgb[:, :, 0], imagen_rgb[:, :, 1])
    dif_rb = cv2.absdiff(imagen_rgb[:, :, 0], imagen_rgb[:, :, 2])
    dif_gb = cv2.absdiff(imagen_rgb[:, :, 1], imagen_rgb[:, :, 2])
    return (
        int(dif_rg.max()) <= tolerancia
        and int(dif_rb.max()) <= tolerancia
        and int(dif_gb.max()) <= tolerancia
    )
