from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .imaging import TITULO_CANAL, matriz_a_qimage


def matriz_a_pixmap(imagen: np.ndarray) -> QPixmap:
    qimage = matriz_a_qimage(imagen)
    return QPixmap.fromImage(qimage)


class EtiquetaImagen(QLabel):
    def __init__(self, texto_vacio: str = "Sin imagen") -> None:
        super().__init__(texto_vacio)
        self._pixmap_base: QPixmap | None = None
        self._texto_vacio = texto_vacio
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setObjectName("imageSurface")

    def fijar_imagen(self, imagen: np.ndarray | None) -> None:
        self._pixmap_base = None if imagen is None else matriz_a_pixmap(imagen)
        if self._pixmap_base is None:
            self.setPixmap(QPixmap())
            self.setText(self._texto_vacio)
            return
        self.setText("")
        self._actualizar_pixmap()

    def resizeEvent(self, event) -> None:  
        super().resizeEvent(event)
        self._actualizar_pixmap()

    def _actualizar_pixmap(self) -> None:
        if self._pixmap_base is None:
            return
        escalado = self._pixmap_base.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(escalado)


class PanelImagen(QFrame):
    def __init__(self, titulo: str, alto_minimo: int = 220) -> None:
        super().__init__()
        self.setObjectName("panelCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.titulo = QLabel(titulo)
        self.titulo.setObjectName("panelTitle")
        layout.addWidget(self.titulo)

        self.imagen = EtiquetaImagen()
        self.imagen.setMinimumHeight(alto_minimo)
        layout.addWidget(self.imagen, 1)

        self.descripcion = QLabel("Esperando datos")
        self.descripcion.setObjectName("panelNote")
        self.descripcion.setWordWrap(True)
        layout.addWidget(self.descripcion)

    def fijar_imagen(self, imagen: np.ndarray | None, descripcion: str = "") -> None:
        self.imagen.fijar_imagen(imagen)
        self.descripcion.setText(descripcion or " ")


class TarjetaDato(QFrame):
    def __init__(self, titulo: str, valor: str = "-") -> None:
        super().__init__()
        self.setObjectName("statTile")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.titulo = QLabel(titulo)
        self.titulo.setObjectName("statTitle")
        layout.addWidget(self.titulo)

        self.valor = QLabel(valor)
        self.valor.setObjectName("statValue")
        self.valor.setWordWrap(True)
        layout.addWidget(self.valor)

    def fijar_valor(self, valor: str) -> None:
        self.valor.setText(valor)


class PanelHistograma(QWidget):
    def __init__(self, color_acento: str) -> None:
        super().__init__()
        self.setMinimumHeight(190)
        self._color_acento = QColor(color_acento)
        self._histograma_base = np.zeros(256, dtype=np.float32)
        self._histograma_ajustado = np.zeros(256, dtype=np.float32)

    def fijar_datos(
        self,
        histograma_base: np.ndarray,
        histograma_ajustado: np.ndarray,
    ) -> None:
        self._histograma_base = histograma_base
        self._histograma_ajustado = histograma_ajustado
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        caja = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(caja, QColor("#0f172a"))
        painter.setPen(QColor("#21314d"))
        painter.drawRect(caja)

        pico = max(
            float(self._histograma_base.max()) if self._histograma_base.size else 0.0,
            float(self._histograma_ajustado.max()) if self._histograma_ajustado.size else 0.0,
            1.0,
        )

        grafica = QRectF(56, 24, max(10, self.width() - 68), max(10, self.height() - 52))
        painter.fillRect(grafica, QColor("#111c32"))

        self._dibujar_histograma(
            painter,
            grafica,
            self._histograma_base,
            pico,
            QColor("#94a3b8"),
            0.32,
        )
        self._dibujar_histograma(
            painter,
            grafica,
            self._histograma_ajustado,
            pico,
            self._color_acento,
            0.90,
        )
        self._dibujar_guias(painter, grafica, pico)

    def _dibujar_histograma(
        self,
        painter: QPainter,
        grafica: QRectF,
        valores: np.ndarray,
        pico: float,
        color: QColor,
        opacidad: float,
    ) -> None:
        path = QPainterPath()
        path.moveTo(grafica.left(), grafica.bottom())
        for indice, cantidad in enumerate(valores):
            x = grafica.left() + (grafica.width() * indice / 255.0)
            normalizado = float(cantidad) / pico
            y = grafica.bottom() - (grafica.height() * normalizado)
            path.lineTo(x, y)
        path.lineTo(grafica.right(), grafica.bottom())
        path.closeSubpath()

        relleno = QColor(color)
        relleno.setAlphaF(opacidad * 0.25)
        borde = QColor(color)
        borde.setAlphaF(opacidad)
        painter.fillPath(path, relleno)
        painter.setPen(QPen(borde, 1.5))
        painter.drawPath(path)

    def _dibujar_guias(self, painter: QPainter, grafica: QRectF, pico: float) -> None:
        painter.setPen(QColor("#4c5b78"))
        for relacion in (0.25, 0.50, 0.75):
            y = grafica.top() + (grafica.height() * relacion)
            painter.drawLine(grafica.left(), y, grafica.right(), y)

        painter.setPen(QColor("#d6e2ff"))
        painter.setFont(QFont("Segoe UI", 8))
        for valor in (0, 64, 128, 192, 255):
            x = grafica.left() + (grafica.width() * valor / 255.0)
            etiqueta = QRectF(x - 18, grafica.bottom() + 2, 36, 14)
            painter.drawText(etiqueta, Qt.AlignHCenter, str(valor))

        # Eje Y: frecuencias absolutas de pixeles.
        for relacion in (1.0, 0.75, 0.50, 0.25, 0.0):
            y = grafica.top() + (grafica.height() * relacion)
            valor = int(round((1.0 - relacion) * pico))
            etiqueta = QRectF(4, y - 8, 48, 16)
            painter.drawText(etiqueta, Qt.AlignRight | Qt.AlignVCenter, str(valor))

        painter.setPen(QColor("#c7d2fe"))
        painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        painter.drawText(QRectF(4, 2, 140, 16), Qt.AlignLeft | Qt.AlignVCenter, "Frecuencia de pixeles")


@dataclass(slots=True)
class FilaDeslizador:
    etiqueta: QLabel
    deslizador: QSlider
    valor: QLabel


class EditorCanal(QFrame):
    nivelCambiado = Signal(str, int)

    def __init__(self, nombre_canal: str, color_acento: str) -> None:
        super().__init__()
        self.nombre_canal = nombre_canal
        self.setObjectName("panelCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        encabezado = QHBoxLayout()
        encabezado.setContentsMargins(0, 0, 0, 0)
        encabezado.setSpacing(8)

        marcador = QFrame()
        marcador.setFixedSize(14, 14)
        marcador.setStyleSheet(
            f"background: {color_acento}; border-radius: 7px; border: 1px solid rgba(0, 0, 0, 0.08);"
        )
        encabezado.addWidget(marcador)

        titulo = QLabel(f"Canal {TITULO_CANAL[nombre_canal]}")
        titulo.setObjectName("panelTitle")
        encabezado.addWidget(titulo, 1)

        self.boton_reiniciar = QPushButton("100%")
        self.boton_reiniciar.setObjectName("tinyButton")
        self.boton_reiniciar.clicked.connect(self._restaurar)
        encabezado.addWidget(self.boton_reiniciar)
        layout.addLayout(encabezado)

        self.histograma = PanelHistograma(color_acento)
        layout.addWidget(self.histograma)

        self.fila_nivel = self._crear_fila_deslizador("Intensidad", 0, 200, 100)
        layout.addLayout(self._crear_layout_fila(self.fila_nivel))

        self.detalle = QLabel("Escala multiplicativa del canal.")
        self.detalle.setObjectName("panelNote")
        self.detalle.setWordWrap(True)
        layout.addWidget(self.detalle)

        self.fila_nivel.deslizador.valueChanged.connect(self._emitir_cambio)

    def valor_actual(self) -> int:
        return self.fila_nivel.deslizador.value()

    def fijar_histogramas(
        self,
        histograma_base: np.ndarray,
        histograma_ajustado: np.ndarray,
    ) -> None:
        self.histograma.fijar_datos(histograma_base, histograma_ajustado)

    def fijar_detalle(self, texto: str) -> None:
        self.detalle.setText(texto)

    def reiniciar(self, emitir_senal: bool = True) -> None:
        self.fila_nivel.deslizador.blockSignals(True)
        self.fila_nivel.deslizador.setValue(100)
        self.fila_nivel.deslizador.blockSignals(False)
        self.fila_nivel.valor.setText("100%")
        if emitir_senal:
            self.nivelCambiado.emit(self.nombre_canal, 100)

    def _restaurar(self) -> None:
        self.reiniciar()

    def _emitir_cambio(self, valor: int) -> None:
        self.fila_nivel.valor.setText(f"{valor}%")
        self.nivelCambiado.emit(self.nombre_canal, valor)

    @staticmethod
    def _crear_fila_deslizador(
        texto: str,
        minimo: int,
        maximo: int,
        inicial: int,
    ) -> FilaDeslizador:
        etiqueta = QLabel(texto)
        etiqueta.setObjectName("fieldLabel")
        deslizador = QSlider(Qt.Horizontal)
        deslizador.setRange(minimo, maximo)
        deslizador.setValue(inicial)
        deslizador.setSingleStep(1)
        valor = QLabel(f"{inicial}%")
        valor.setAlignment(Qt.AlignCenter)
        valor.setObjectName("valueBadge")
        valor.setMinimumWidth(58)
        return FilaDeslizador(etiqueta=etiqueta, deslizador=deslizador, valor=valor)

    @staticmethod
    def _crear_layout_fila(fila: FilaDeslizador) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(fila.etiqueta)
        layout.addWidget(fila.deslizador, 1)
        layout.addWidget(fila.valor)
        return layout
