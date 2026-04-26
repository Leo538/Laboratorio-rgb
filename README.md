# Laboratorio RGB - Procesamiento de Imagenes

Aplicacion de escritorio en Python para analizar y ajustar imagenes por canales RGB, con visualizacion de histogramas y salidas derivadas.

## Que hace

- Carga una imagen RGB desde disco.
- Separa y muestra los canales `R`, `G` y `B`.
- Permite ajustar la intensidad de cada canal con sliders.
- Muestra histogramas de canal (original vs ajustado).
- Genera salidas:
  - Imagen nueva ajustada (mezcla RGB).
  - Imagen recortada.
  - Imagen binaria (blanco y negro con umbral).
- Permite guardar cada salida en la carpeta `salidas/`.

## Estructura del proyecto

- `main.py`: punto de entrada de la aplicacion.
- `laboratorio_rgb/`: paquete principal de la interfaz y logica.
  - `window.py`: ventana principal y flujo de UI.
  - `widgets.py`: componentes visuales personalizados.
  - `imaging.py`: operaciones de procesamiento de imagen.
- `requirements.txt`: dependencias.
- `.gitignore`: exclusiones para Git.

## Requisitos

- Python 3.10+ (recomendado)
- Dependencias de `requirements.txt`:
  - `numpy`
  - `opencv-python`
  - `PySide6`

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

Desde la raiz del proyecto:

```bash
python main.py
```

En Windows, si usas una ruta especifica de Python:

```powershell
& "C:/Program Files/Python314/python.exe" "main.py"
```

## Flujo de uso rapido

1. Haz clic en **Abrir imagen**.
2. Ajusta intensidad de canales RGB en los sliders.
3. Configura **Recorte** y **Umbral B/N**.
4. Revisa salidas en los paneles inferiores.
5. Guarda con:
   - **Guardar nueva**
   - **Guardar recortada**
   - **Guardar B/N**

## Notas

- Las salidas se proponen por defecto dentro de `salidas/`.
- Si la carpeta no existe, la aplicacion la crea automaticamente.
