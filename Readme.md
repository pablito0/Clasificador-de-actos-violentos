# Clasificador de Actos Violentos

Este proyecto consta de dos partes principales: un autoencoder y un clasificador. En las secciones correspondientes se explica cómo utilizar cada modelo.

## Contenido
- [Instalación](#instalación)
- [Uso](#uso)
  - [Clasificador](#clasificador)
  - [Autoencoder](#autoencoder)
- [Dependencias](#dependencias)
- [Licencia](#licencia)

## Instalación

Sigue las instrucciones para configurar el entorno y instalar todas las dependencias necesarias para ejecutar el proyecto.

## Uso

### Clasificador

Para probar el clasificador:
1. Ejecuta el archivo `preprocess.py` que se encuentra en la carpeta del clasificador. Esto preparará los datos y generará una nueva estructura en la carpeta `data`.
2. Para entrenar y testear el clasificador, ejecuta `classificador.py` pasándole por parámetro la ruta de la carpeta `data`.

### Autoencoder

Para probar el autoencoder:
1. Ejecuta `preprocess.py` dentro de la carpeta del autoencoder, pasándole por argumento la ruta de los vídeos del dataset descargado.
2. Ejecuta `train.py` para entrenar el modelo y `test.py`y para probarlo.

## Dependencias

Para ejecutar el proyecto completo, necesitarás las siguientes librerías:
- numpy
- sklearn
- keras
- tensorflow
- FFMPEG
- scipy
- OpenCV

Dependencias adicionales de los datasets:
- **Autoencoder**: Avenue Dataset disponible en [este enlace](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html).
- **Clasificador**: UCF Crime dataset, que se puede encontrar en [este enlace](https://www.crcv.ucf.edu/projects/real-world/).

## Licencia
Esta obra está sujeta a una licencia de Reconocimiento-NoComercial-SinObraDerivada 3.0 España de Creative Commons 
