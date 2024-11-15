
# Recopilación de Datos para Entrenamiento y Calibración en Seguimiento de la Mirada

## Instrucciones para ejecutar

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Calibra la cámara utilizando el script interactivo proporcionado:
   ```bash
   python camera_calibration.py
   ```
   Para la extracción se abrirá una ventana con la imagen de la cámara, y se tomarán capturas con Espacio o Enter. Para poder realizar una calibracion se rerquieren un minimo de 15, pero es recomendable usar entre 20 y 25 imágenes. Para una mejor calibración se recomienda ubicar el chess-grid en diferentes zonas, distancias y angulos, para abordar el mayor espacio posible de la imagen.
   Puedes consultar más detalles sobre la calibración de la cámara en la documentación oficial de [OpenCV](https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html).

3. Ejecuta el script principal para iniciar la recopilación de datos:
   ```bash
   python camera_calibration.py --base_path=./data
   ```
   Este código ha sido probado en Ubuntu 22.04 Si usas macOS o Windows, puede que necesites proporcionar manualmente los parámetros del monitor debido a la compleja instalación de la librería pgi. Ejemplo:
   ```bash
   --monitor_mm=750,420 --monitor_pixels=1920,1080
   ```
   Además, es posible que necesites ajustar los valores de `TargetOrientation` en el archivo `utils.py`.

4. Durante la recopilación de datos, mira la pantalla y presiona la tecla de flecha correspondiente a la dirección en la que apunta la letra `E` cuando su color cambie de azul a naranja. Es recomendable presionar la tecla varias veces ya que OpenCV a veces no registra el primer clic.

5. Presiona la tecla `q` cuando la recopilación de datos haya finalizado.

6. Puedes visualizar los datos recopilados imagen por imagen en 3D ejecutando el siguiente comando:
   ```bash
   python visualization.py --base_path=./data
   ```

**Créditos**: Parte de este código código fue desarrollado originalmente por [P. Perle](https://github.com/pperle) como parte de su tesis de maestría. La herramienta de extracción de datos para el fine.tuning ha sido adaptada con algunas modificaciones para ajustarse a las necesidades específicas de este proyecto.
Este código es complementario al proyecto, pero no necesario. Su utilización supone incrementar los datos de entrenamiento para maximizar la predicción del modelo sin la necesidad de calibración del plano de la pantalla respecto la cámara y de `TargetOrientation`.