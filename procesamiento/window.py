from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .imaging import (
    COLOR_CANAL,
    ORDEN_CANALES,
    TITULO_CANAL,
    PerfilMezcla,
    abrir_imagen_rgb,
    aplicar_perfil_rgb,
    binarizar_imagen,
    calcular_histograma,
    combinar_canales_rgb,
    convertir_a_grises,
    describir_canal,
    guardar_imagen,
    imagen_es_gris,
    separar_canales_rgb,
)
from .widgets import EditorCanal, PanelImagen, TarjetaDato


class VentanaLaboratorioRGB(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Procesamiento de imágenes")
        self.resize(1500, 940)
        self.setMinimumSize(1160, 760)

        self.ruta_imagen: Path | None = None
        self.imagen_original: np.ndarray | None = None
        self.canales_originales: dict[str, np.ndarray] = {}
        self.histogramas_originales: dict[str, np.ndarray] = {}
        self.canales_ajustados: dict[str, np.ndarray] = {}
        self.imagen_mezclada: np.ndarray | None = None
        self.imagen_mascara: np.ndarray | None = None
        self.imagen_binaria: np.ndarray | None = None
        self._bloquear_actualizacion = False

        self.temporizador = QTimer(self)
        self.temporizador.setSingleShot(True)
        self.temporizador.setInterval(35)
        self.temporizador.timeout.connect(self.procesar_imagen)

        self._construir_interfaz()
        self._aplicar_estilos()
        self._configurar_barra_estado()
        self._activar_controles(False)
        self._informar("Interfaz lista. Abre una imagen para comenzar.")

    def _construir_interfaz(self) -> None:
        contenedor = QWidget()
        layout_principal = QVBoxLayout(contenedor)
        layout_principal.setContentsMargins(22, 22, 22, 22)
        layout_principal.setSpacing(18)
        self.setCentralWidget(contenedor)

        layout_principal.addWidget(self._crear_encabezado())

        self.area_scroll = QScrollArea()
        self.area_scroll.setWidgetResizable(True)
        self.area_scroll.setFrameShape(QFrame.NoFrame)
        self.area_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.area_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.area_scroll.verticalScrollBar().setSingleStep(18)
        self.area_scroll.verticalScrollBar().setPageStep(120)
        layout_principal.addWidget(self.area_scroll, 1)

        contenido = QWidget()
        self.area_scroll.setWidget(contenido)
        layout_contenido = QVBoxLayout(contenido)
        layout_contenido.setContentsMargins(0, 0, 4, 0)
        layout_contenido.setSpacing(18)

        layout_contenido.addWidget(self._crear_etiqueta_seccion("Vista principal"))
        fila_principal = QHBoxLayout()
        fila_principal.setSpacing(12)
        self.panel_original = PanelImagen("Imagen original", alto_minimo=230)
        self.panel_mezcla = PanelImagen("Imagen nueva ajustada", alto_minimo=230)
        fila_principal.addWidget(self.panel_original, 1)
        fila_principal.addWidget(self.panel_mezcla, 1)
        layout_contenido.addLayout(fila_principal)

        layout_contenido.addWidget(self._crear_etiqueta_seccion("Canales RGB originales"))
        fila_canales_base = QHBoxLayout()
        fila_canales_base.setSpacing(12)
        self.paneles_base: dict[str, PanelImagen] = {}
        for nombre_canal in ORDEN_CANALES:
            panel = PanelImagen(f"Canal {nombre_canal.lower()} base", alto_minimo=190)
            fila_canales_base.addWidget(panel, 1)
            self.paneles_base[nombre_canal] = panel
        layout_contenido.addLayout(fila_canales_base)

        layout_contenido.addWidget(self._crear_etiqueta_seccion("Histogramas y control de intensidad"))
        fila_editores = QHBoxLayout()
        fila_editores.setSpacing(12)
        self.editores_canal: dict[str, EditorCanal] = {}
        for nombre_canal in ORDEN_CANALES:
            editor = EditorCanal(nombre_canal, COLOR_CANAL[nombre_canal])
            editor.nivelCambiado.connect(self._cuando_cambia_nivel)
            fila_editores.addWidget(editor, 1)
            self.editores_canal[nombre_canal] = editor
        layout_contenido.addLayout(fila_editores)

        layout_contenido.addWidget(self._crear_etiqueta_seccion("Canales RGB ajustados"))
        fila_canales_ajustados = QHBoxLayout()
        fila_canales_ajustados.setSpacing(12)
        self.paneles_ajustados: dict[str, PanelImagen] = {}
        for nombre_canal in ORDEN_CANALES:
            panel = PanelImagen(f"Canal {nombre_canal.lower()} ajustado", alto_minimo=190)
            fila_canales_ajustados.addWidget(panel, 1)
            self.paneles_ajustados[nombre_canal] = panel
        layout_contenido.addLayout(fila_canales_ajustados)

        layout_contenido.addWidget(self._crear_etiqueta_seccion("Mascara espacial y binarizacion"))
        layout_contenido.addWidget(self._crear_panel_controles())

        fila_salida = QHBoxLayout()
        fila_salida.setSpacing(12)
        self.panel_mascara = PanelImagen("Imagen recortada", alto_minimo=190)
        self.panel_binaria = PanelImagen("Imagen blanco y negro", alto_minimo=190)
        fila_salida.addWidget(self.panel_mascara, 1)
        fila_salida.addWidget(self.panel_binaria, 1)
        layout_contenido.addLayout(fila_salida)
        self.panel_resumen = self._crear_panel_resumen()
        self.panel_resumen.hide()

    def _crear_encabezado(self) -> QFrame:
        tarjeta = QFrame()
        tarjeta.setObjectName("heroCard")

        layout = QVBoxLayout(tarjeta)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        fila_superior = QHBoxLayout()
        fila_superior.setSpacing(18)
        layout.addLayout(fila_superior)

        columna_texto = QVBoxLayout()
        columna_texto.setSpacing(8)
        fila_superior.addLayout(columna_texto, 1)

        titulo = QLabel("Procesamiento de imágenes")
        titulo.setObjectName("heroTitle")
        columna_texto.addWidget(titulo)

        subtitulo = QLabel(
            "Carga una imagen, separa sus canales rojo, verde y azul, modifica cada uno "
            "con sliders y observa el histograma antes y despues del ajuste."
        )
        subtitulo.setObjectName("heroSubtitle")
        subtitulo.setWordWrap(True)
        columna_texto.addWidget(subtitulo)

        fila_insignias = QHBoxLayout()
        fila_insignias.setSpacing(10)
        fila_insignias.addWidget(self._crear_insignia("Canal rojo", "#33141c", "#fda4af"))
        fila_insignias.addWidget(self._crear_insignia("Canal verde", "#0f2d24", "#86efac"))
        fila_insignias.addWidget(self._crear_insignia("Canal azul", "#10233a", "#93c5fd"))
        fila_insignias.addStretch(1)
        columna_texto.addLayout(fila_insignias)

        grilla_acciones = QGridLayout()
        grilla_acciones.setHorizontalSpacing(10)
        grilla_acciones.setVerticalSpacing(10)
        fila_superior.addLayout(grilla_acciones)

        self.boton_abrir = QPushButton("Abrir imagen")
        self.boton_abrir.setObjectName("actionButton")
        self.boton_abrir.clicked.connect(self.cargar_imagen)
        self.boton_abrir.setShortcut("Ctrl+O")
        grilla_acciones.addWidget(self.boton_abrir, 0, 0)

        self.boton_reiniciar = QPushButton("Reiniciar")
        self.boton_reiniciar.setObjectName("secondaryAction")
        self.boton_reiniciar.clicked.connect(self.reiniciar_controles)
        self.boton_reiniciar.setShortcut("Ctrl+R")
        grilla_acciones.addWidget(self.boton_reiniciar, 0, 1)

        self.boton_guardar_mezcla = QPushButton("Guardar nueva")
        self.boton_guardar_mezcla.setObjectName("secondaryAction")
        self.boton_guardar_mezcla.clicked.connect(lambda: self.guardar_resultado("mezcla"))
        self.boton_guardar_mezcla.setShortcut("Ctrl+S")
        grilla_acciones.addWidget(self.boton_guardar_mezcla, 1, 0)

        self.boton_guardar_mascara = QPushButton("Guardar recortada")
        self.boton_guardar_mascara.setObjectName("secondaryAction")
        self.boton_guardar_mascara.clicked.connect(lambda: self.guardar_resultado("mascara"))
        grilla_acciones.addWidget(self.boton_guardar_mascara, 1, 1)

        self.boton_guardar_binaria = QPushButton("Guardar B/N")
        self.boton_guardar_binaria.setObjectName("secondaryAction")
        self.boton_guardar_binaria.clicked.connect(lambda: self.guardar_resultado("binaria"))
        self.boton_guardar_binaria.setShortcut("Ctrl+Shift+S")
        grilla_acciones.addWidget(self.boton_guardar_binaria, 2, 0, 1, 2)

        fila_datos = QHBoxLayout()
        fila_datos.setSpacing(14)
        layout.addLayout(fila_datos)

        self.tarjeta_archivo = TarjetaDato("Archivo", "Sin cargar")
        self.tarjeta_tamano = TarjetaDato("Tamano", "-")
        self.tarjeta_mascara = TarjetaDato("Recorte", "8%")
        self.tarjeta_tamano_salida = TarjetaDato("Tamano salida", "-")

        fila_datos.addWidget(self.tarjeta_archivo, 2)
        fila_datos.addWidget(self.tarjeta_tamano, 1)
        fila_datos.addWidget(self.tarjeta_mascara, 1)
        fila_datos.addWidget(self.tarjeta_tamano_salida, 1)
        return tarjeta

    def _crear_panel_controles(self) -> QFrame:
        tarjeta = QFrame()
        tarjeta.setObjectName("insightCard")

        layout = QVBoxLayout(tarjeta)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        titulo = QLabel("Controles posteriores")
        titulo.setObjectName("helperTitle")
        layout.addWidget(titulo)

        descripcion = QLabel(
            "La mascara espacial reduce la resolucion de la imagen y luego la expande para "
            "generar bloques. Despues, la salida se convierte a gris y se binariza."
        )
        descripcion.setObjectName("helperBody")
        descripcion.setWordWrap(True)
        layout.addWidget(descripcion)

        fila_doble = QHBoxLayout()
        fila_doble.setContentsMargins(0, 0, 0, 0)
        fila_doble.setSpacing(14)

        caja_mascara = QFrame()
        caja_mascara.setObjectName("statTile")
        layout_mascara = QVBoxLayout(caja_mascara)
        layout_mascara.setContentsMargins(12, 12, 12, 12)
        layout_mascara.setSpacing(8)
        titulo_mascara = QLabel("Mascara")
        titulo_mascara.setObjectName("helperTitle")
        layout_mascara.addWidget(titulo_mascara)

        self.deslizador_recorte, self.etiqueta_recorte = self._crear_deslizador_lateral(
            layout_mascara,
            "Recorte",
            1,
            100,
            8,
            self._cuando_cambia_mascara,
        )
        fila_doble.addWidget(caja_mascara, 1)

        caja_binaria = QFrame()
        caja_binaria.setObjectName("statTile")
        layout_binaria = QVBoxLayout(caja_binaria)
        layout_binaria.setContentsMargins(12, 12, 12, 12)
        layout_binaria.setSpacing(8)
        titulo_binaria = QLabel("Binarizacion")
        titulo_binaria.setObjectName("helperTitle")
        layout_binaria.addWidget(titulo_binaria)

        self.deslizador_umbral, self.etiqueta_umbral = self._crear_deslizador_lateral(
            layout_binaria,
            "Umbral B/N",
            0,
            255,
            128,
            self._cuando_cambia_umbral,
        )
        fila_doble.addWidget(caja_binaria, 1)

        layout.addLayout(fila_doble)

        self.etiqueta_estado = QLabel("Abre una imagen para comenzar.")
        self.etiqueta_estado.setObjectName("statusText")
        self.etiqueta_estado.setWordWrap(True)
        layout.addWidget(self.etiqueta_estado)
        return tarjeta

    def _crear_panel_resumen(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("insightCard")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        titulo = QLabel("Resumen del procesamiento")
        titulo.setObjectName("helperTitle")
        layout.addWidget(titulo)

        self.etiqueta_resumen = QLabel(
            "Aqui se mostrara una descripcion corta del flujo aplicado a la imagen."
        )
        self.etiqueta_resumen.setObjectName("helperBody")
        self.etiqueta_resumen.setWordWrap(True)
        layout.addWidget(self.etiqueta_resumen)

        layout.addWidget(
            self._crear_insignia(
                "Flujo: cargar -> separar -> ajustar -> recomponer -> mascara -> binarizar",
                "#2b1f0f",
                "#facc15",
            )
        )
        return panel

    def _aplicar_estilos(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            """
            QWidget {
                background: #0b1220;
                color: #dbe7ff;
                font-size: 10.5pt;
            }
            QMainWindow {
                background: #0b1220;
            }
            #heroCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #111b31,
                    stop: 0.5 #0f1a2f,
                    stop: 1 #14213d
                );
                border: 1px solid #233451;
                border-radius: 24px;
            }
            #panelCard, #insightCard, #statTile {
                background: #101b2f;
                border: 1px solid #20324f;
                border-radius: 18px;
            }
            #heroTitle {
                font-size: 22pt;
                font-weight: 800;
                color: #f8fbff;
                letter-spacing: 0.03em;
            }
            #heroSubtitle {
                color: #9fb5d6;
                line-height: 1.35em;
            }
            #panelTitle, #helperTitle {
                font-size: 12pt;
                font-weight: 700;
                color: #f3f7ff;
            }
            #sectionLabel {
                font-size: 11pt;
                font-weight: 700;
                color: #67e8f9;
                padding-left: 2px;
            }
            #panelNote, #fieldLabel, #helperBody {
                color: #98acc9;
            }
            #statusText {
                background: #0c1a2e;
                border: 1px solid #2c4368;
                border-radius: 12px;
                color: #dbe7ff;
                padding: 10px 12px;
            }
            #imageSurface {
                background: #0a172a;
                border: 1px dashed #37557e;
                border-radius: 14px;
                color: #87a3ca;
                padding: 12px;
            }
            #actionButton, #secondaryAction, #tinyButton {
                min-height: 42px;
                border-radius: 12px;
                padding: 0 16px;
                font-weight: 700;
            }
            #actionButton {
                background: #06b6d4;
                color: #062029;
                border: 1px solid #67e8f9;
            }
            #actionButton:hover {
                background: #22d3ee;
            }
            #secondaryAction, #tinyButton {
                background: #152540;
                color: #dbe7ff;
                border: 1px solid #355784;
            }
            #secondaryAction:hover, #tinyButton:hover {
                background: #1c3155;
            }
            #valueBadge {
                background: #1d4ed8;
                color: #eff6ff;
                border-radius: 10px;
                padding: 6px 8px;
                font-weight: 700;
            }
            #statTitle {
                color: #8ba4c7;
                font-size: 8.8pt;
                font-weight: 700;
                text-transform: uppercase;
            }
            #statValue {
                color: #e2ecff;
                font-size: 11.5pt;
                font-weight: 700;
            }
            QSlider::groove:horizontal {
                background: #243a5c;
                height: 7px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #22d3ee;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e0f2fe;
                border: 2px solid #22d3ee;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QStatusBar {
                background: #0f1a2f;
                border-top: 1px solid #223656;
                color: #9eb3d1;
                padding: 4px 10px;
            }
            """
        )

    def _configurar_barra_estado(self) -> None:
        barra = QStatusBar(self)
        barra.setSizeGripEnabled(False)
        self.setStatusBar(barra)

    def _activar_controles(self, activo: bool) -> None:
        for widget in (
            self.boton_reiniciar,
            self.boton_guardar_mezcla,
            self.boton_guardar_mascara,
            self.boton_guardar_binaria,
            self.deslizador_recorte,
            self.deslizador_umbral,
        ):
            widget.setEnabled(activo)

        for editor in self.editores_canal.values():
            editor.setEnabled(activo)

    def cargar_imagen(self) -> None:
        carpeta_base = self.ruta_imagen.parent if self.ruta_imagen else Path.cwd()
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona una imagen",
            str(carpeta_base),
            "Imagenes (*.png *.jpg *.jpeg *.bmp)",
        )
        if not ruta:
            return

        try:
            imagen = abrir_imagen_rgb(ruta)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error de carga", str(exc))
            return

        self.ruta_imagen = Path(ruta)
        self.imagen_original = imagen
        self.canales_originales = separar_canales_rgb(imagen)
        self.histogramas_originales = {
            nombre: calcular_histograma(canal)
            for nombre, canal in self.canales_originales.items()
        }

        self._bloquear_actualizacion = True
        for editor in self.editores_canal.values():
            editor.reiniciar(emitir_senal=False)
        self._reiniciar_controles_posteriores()
        self._bloquear_actualizacion = False

        self.tarjeta_archivo.fijar_valor(self.ruta_imagen.name)
        self.tarjeta_tamano.fijar_valor(f"{imagen.shape[1]} x {imagen.shape[0]} px")
        self.panel_original.fijar_imagen(
            imagen,
            f"Base cargada | promedio general = {float(imagen.mean()):.1f}",
        )
        for nombre_canal in ORDEN_CANALES:
            canal = self.canales_originales[nombre_canal]
            self.paneles_base[nombre_canal].fijar_imagen(
                canal,
                f"{TITULO_CANAL[nombre_canal]} base | {describir_canal(canal)}",
            )

        self._activar_controles(True)
        if imagen_es_gris(imagen):
            self._informar(
                "La imagen cargada es monocromatica. Los canales RGB se veran en escala de grises."
            )
        else:
            self._informar("Imagen cargada. Ajusta los sliders para modificar cada canal.")
        self.procesar_imagen()

    def reiniciar_controles(self) -> None:
        if self.imagen_original is None:
            return

        self._bloquear_actualizacion = True
        for editor in self.editores_canal.values():
            editor.reiniciar(emitir_senal=False)
        self._reiniciar_controles_posteriores()
        self._bloquear_actualizacion = False

        self._informar("Se restauraron los valores base.")
        self.procesar_imagen()

    def guardar_resultado(self, variante: str) -> None:
        mapa_imagenes = {
            "mezcla": self.imagen_mezclada,
            "mascara": self.imagen_mascara,
            "binaria": self.imagen_binaria,
        }
        imagen = mapa_imagenes[variante]
        if imagen is None:
            QMessageBox.warning(self, "Sin imagen", "Todavia no hay datos para guardar.")
            return

        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar resultado",
            str(self._sugerir_nombre(variante)),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not ruta:
            return

        try:
            guardar_imagen(ruta, imagen)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error al guardar", str(exc))
            return

        self._informar(f"Se guardo {Path(ruta).name}")

    def _sugerir_nombre(self, variante: str) -> Path:
        base = self.ruta_imagen.stem if self.ruta_imagen else "resultado"
        carpeta = self._carpeta_salidas()
        sufijos = {
            "mezcla": "_nueva_rgb.png",
            "mascara": "_recortada.png",
            "binaria": "_bn.png",
        }
        return carpeta / f"{base}{sufijos[variante]}"

    @staticmethod
    def _carpeta_salidas() -> Path:
        # Guarda todos los resultados dentro del proyecto.
        carpeta = Path(__file__).resolve().parent.parent / "salidas"
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    def _cuando_cambia_nivel(self, nombre_canal: str, nivel: int) -> None:
        del nombre_canal, nivel
        if self.imagen_original is None or self._bloquear_actualizacion:
            return
        self._informar("Recalculando los tres canales y sus histogramas...")
        self.temporizador.start()

    def _cuando_cambia_umbral(self, valor: int) -> None:
        self.etiqueta_umbral.setText(str(valor))
        if self.imagen_original is None or self._bloquear_actualizacion:
            return
        self._informar("Actualizando binarizacion blanco y negro...")
        self.temporizador.start()

    def _cuando_cambia_mascara(self, _valor: int) -> None:
        valor = self.deslizador_recorte.value()
        self.etiqueta_recorte.setText(str(valor))
        self.tarjeta_mascara.fijar_valor(f"{valor}%")
        if self.imagen_original is None or self._bloquear_actualizacion:
            return
        self._informar("Aplicando mascara espacial...")
        self.temporizador.start()

    def procesar_imagen(self) -> None:
        if self.imagen_original is None:
            return

        perfil = PerfilMezcla(
            nivel_rojo=self.editores_canal["R"].valor_actual(),
            nivel_verde=self.editores_canal["G"].valor_actual(),
            nivel_azul=self.editores_canal["B"].valor_actual(),
        )

        self.canales_ajustados = aplicar_perfil_rgb(self.canales_originales, perfil)
        self.imagen_mezclada = combinar_canales_rgb(self.canales_ajustados)
        recorte = self.deslizador_recorte.value()
        self.imagen_mascara = self._aplicar_recorte_central(self.imagen_mezclada, recorte)
        gris = convertir_a_grises(self.imagen_mascara)
        self.imagen_binaria = binarizar_imagen(gris, self.deslizador_umbral.value())

        self.panel_mezcla.fijar_imagen(
            self.imagen_mezclada,
            (
                f"Combinacion RGB | R {perfil.nivel_rojo}% | "
                f"G {perfil.nivel_verde}% | B {perfil.nivel_azul}%"
            ),
        )
        self.panel_mascara.fijar_imagen(
            self.imagen_mascara,
            (
                f"Recorte {recorte}% | "
                f"{self.imagen_mascara.shape[1]} x {self.imagen_mascara.shape[0]} px"
            ),
        )
        self.panel_binaria.fijar_imagen(
            self.imagen_binaria,
            (
                f"Binaria con umbral {self.deslizador_umbral.value()} | "
                f"media={float(self.imagen_binaria.mean()):.1f}"
            ),
        )

        for nombre_canal in ORDEN_CANALES:
            base = self.canales_originales[nombre_canal]
            ajustado = self.canales_ajustados[nombre_canal]
            self.paneles_ajustados[nombre_canal].fijar_imagen(
                ajustado,
                (
                    f"{TITULO_CANAL[nombre_canal]} al "
                    f"{self.editores_canal[nombre_canal].valor_actual()}% | "
                    f"{describir_canal(ajustado)}"
                ),
            )
            self.editores_canal[nombre_canal].fijar_histogramas(
                self.histogramas_originales[nombre_canal],
                calcular_histograma(ajustado),
            )
            self.editores_canal[nombre_canal].fijar_detalle(
                f"Original: {describir_canal(base)} | Ajustado: {describir_canal(ajustado)}"
            )

        self.tarjeta_tamano_salida.fijar_valor(
            f"{self.imagen_mascara.shape[1]} x {self.imagen_mascara.shape[0]} px"
        )
        self.etiqueta_resumen.setText(
            (
                f"Se aplico una mezcla RGB con niveles R={perfil.nivel_rojo}%, "
                f"G={perfil.nivel_verde}%, B={perfil.nivel_azul}%, despues un recorte "
                f"de valor {recorte}, "
                f"y finalmente una binarizacion con umbral {self.deslizador_umbral.value()}."
            )
        )
        self._informar("Proceso actualizado correctamente.")

    def _reiniciar_controles_posteriores(self) -> None:
        self.deslizador_recorte.blockSignals(True)
        self.deslizador_umbral.blockSignals(True)
        self.deslizador_recorte.setValue(8)
        self.deslizador_umbral.setValue(128)
        self.deslizador_recorte.blockSignals(False)
        self.deslizador_umbral.blockSignals(False)
        self.etiqueta_recorte.setText("8")
        self.etiqueta_umbral.setText("128")
        self.tarjeta_mascara.fijar_valor("8%")

    @staticmethod
    def _aplicar_recorte_central(imagen: np.ndarray, nivel_recorte: int) -> np.ndarray:
        alto, ancho = imagen.shape[:2]
        # 1 mantiene casi todo; 100 conserva alrededor del 15% central.
        proporcion = 1.0 - ((max(1, min(100, nivel_recorte)) - 1) / 99.0) * 0.85
        nuevo_ancho = max(1, int(round(ancho * proporcion)))
        nuevo_alto = max(1, int(round(alto * proporcion)))

        x0 = max(0, (ancho - nuevo_ancho) // 2)
        y0 = max(0, (alto - nuevo_alto) // 2)
        x1 = min(ancho, x0 + nuevo_ancho)
        y1 = min(alto, y0 + nuevo_alto)
        return imagen[y0:y1, x0:x1].copy()

    def _informar(self, mensaje: str) -> None:
        self.etiqueta_estado.setText(mensaje)
        if self.statusBar() is not None:
            self.statusBar().showMessage(mensaje)

    @staticmethod
    def _crear_etiqueta_seccion(texto: str) -> QLabel:
        etiqueta = QLabel(texto)
        etiqueta.setObjectName("sectionLabel")
        return etiqueta

    @staticmethod
    def _crear_insignia(texto: str, fondo: str, frente: str) -> QLabel:
        insignia = QLabel(texto)
        insignia.setStyleSheet(
            f"""
            QLabel {{
                background: {fondo};
                color: {frente};
                border-radius: 11px;
                padding: 7px 12px;
                font-weight: 700;
            }}
            """
        )
        return insignia

    @staticmethod
    def _crear_deslizador_lateral(
        layout_padre: QVBoxLayout,
        texto: str,
        minimo: int,
        maximo: int,
        inicial: int,
        callback,
    ) -> tuple[QSlider, QLabel]:
        etiqueta = QLabel(texto)
        etiqueta.setObjectName("fieldLabel")
        layout_padre.addWidget(etiqueta)

        fila = QHBoxLayout()
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(10)

        deslizador = QSlider(Qt.Horizontal)
        deslizador.setRange(minimo, maximo)
        deslizador.setValue(inicial)
        deslizador.valueChanged.connect(callback)
        fila.addWidget(deslizador, 1)

        valor = QLabel(str(inicial))
        valor.setObjectName("valueBadge")
        valor.setAlignment(Qt.AlignCenter)
        valor.setMinimumWidth(62)
        fila.addWidget(valor)
        layout_padre.addLayout(fila)
        return deslizador, valor


def iniciar_aplicacion() -> int:
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    ventana = VentanaLaboratorioRGB()
    ventana.show()
    return app.exec()
