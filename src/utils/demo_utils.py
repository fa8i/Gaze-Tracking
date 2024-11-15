import cv2
import pandas as pd
import numpy as np


def create_gaussian_kernel(size, sigma):
    """Crea un kernel gaussiano 2D.

    Args:
        size (int): El tamaño del kernel (debe ser un número impar).
        sigma (float): La desviación estándar de la distribución gaussiana.

    Returns:
        np.ndarray: Kernel gaussiano 2D de tamaño (size, size).
    """
    kx = cv2.getGaussianKernel(size, sigma)
    ky = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(kx, ky)
    return kernel

def create_custom_colormap():
    """"Crea un mapa de colores personalizado.

    Returns:
        np.ndarray: Un array 2D de forma (256, 1, 3), que representa un mapa de colores donde cada índice corresponde a un color en formato RGB.
    """
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
    positions = [0, 51, 102, 153, 204, 255]
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(len(positions)-1):
            if positions[j] <= i <= positions[j+1]:
                ratio = (i - positions[j]) / (positions[j+1] - positions[j])
                colormap[i, 0, 0] = int(colors[j][0] + ratio * (colors[j+1][0] - colors[j][0]))
                colormap[i, 0, 1] = int(colors[j][1] + ratio * (colors[j+1][1] - colors[j][1]))
                colormap[i, 0, 2] = int(colors[j][2] + ratio * (colors[j+1][2] - colors[j][2]))
                break
    return colormap


def predict_point_on_screen(gaze_vector, point_on_screen, rvec, tvec, scaler, regressor, monitor_pixels):
    """Predice el punto en pantalla usando el modelo entrenado (regressor) para corregir la predicción existente.
    
    Args:
        gaze_vector (np.ndarray): Vector de mirada (gaze_pitch, gaze_yaw).
        point_on_screen (tuple): Punto en pantalla (x, y) predicho por el modelo existente.
        rvec (np.ndarray): Vector de rotación.
        tvec (np.ndarray): Vector de traslación.
        scaler (RobustScaler): Scaler cargado.
        regressor: Modelo de regresión entrenado para corregir las predicciones.
        monitor_pixels (tuple): Dimensiones de la pantalla en píxeles (width, height).

    Returns:
        tuple: Coordenadas predichas en pantalla (x, y) corregidas como enteros.
    """
    gaze_pitch, gaze_yaw = gaze_vector[0], gaze_vector[1]
    point_on_screen_x, point_on_screen_y = point_on_screen[0], point_on_screen[1]
    rvec_x, rvec_y, rvec_z = rvec[0], rvec[1], rvec[2]
    tvec_x, tvec_y, tvec_z = tvec[0], tvec[1], tvec[2]

    # Normalizar las características de entrada que no se escalan
    point_on_screen_x_norm = point_on_screen_x / monitor_pixels[0]
    point_on_screen_y_norm = point_on_screen_y / monitor_pixels[1]
    tvec_x_norm = tvec_x / 1000
    tvec_y_norm = tvec_y / 1000
    tvec_z_norm = tvec_z / 1000

    # Crear un array con solo las características que deben ser escaladas
    features_to_scale = np.array([gaze_pitch, gaze_yaw, rvec_x, rvec_y, rvec_z]).reshape(1, -1)
    features_df = pd.DataFrame(features_to_scale, columns=['gaze_pitch', 'gaze_yaw', 'rvec_x', 'rvec_y', 'rvec_z'])
    features_scaled = scaler.transform(features_df)

    # Crear un array final combinando las características escaladas y las ya normalizadas
    input_features = np.hstack([features_scaled, [[point_on_screen_x_norm, point_on_screen_y_norm, tvec_x_norm, tvec_y_norm, tvec_z_norm]]])

    prediction_norm = regressor.predict(input_features)
    predicted_x = prediction_norm[0][0] * monitor_pixels[0]
    predicted_y = prediction_norm[0][1] * monitor_pixels[1]
    predicted_point = (int(round(predicted_x)), int(round(predicted_y)))

    return predicted_point

def calc_heatmap(gaze_x, gaze_y, monitor_pixels, heatmap, colormap):
    """Calcula un heatmap a partir de las coordenadas de la mirada y lo mezcla con un colormap personalizado.

    Args:
        gaze_x (int): Coordenada X del punto de mirada en píxeles.
        gaze_y (int): Coordenada Y del punto de mirada en píxeles.
        monitor_pixels (tuple): Dimensiones del monitor en píxeles (ancho, alto).
        heatmap (np.ndarray): Array 2D que representa el heatmap acumulativo sobre el que se añadirá el nuevo kernel.
        colormap (np.ndarray): Mapa de colores personalizado de tamaño (256, 1, 3) para aplicar al heatmap.

    Returns:
        tuple:
            np.ndarray: Heatmap coloreado de tamaño (alto, ancho, 3), donde cada píxel tiene valores RGB basados en el heatmap original y el colormap.
            np.ndarray: Canal alfa de tamaño (alto, ancho, 1) que representa la transparencia del heatmap, basado en la intensidad normalizada.
    """
    kernel_size = 551  # Debe ser impar
    kernel = create_gaussian_kernel(size=kernel_size, sigma=80)

    # Calcular coordenadas para colocar el kernel
    x1 = gaze_x - kernel_size // 2
    y1 = gaze_y - kernel_size // 2
    x2 = x1 + kernel_size
    y2 = y1 + kernel_size

    # Ajustar índices dentro de los límites
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(monitor_pixels[0], x2)
    y2_clip = min(monitor_pixels[1], y2)
    kx1 = x1_clip - x1
    ky1 = y1_clip - y1
    kx2 = kernel_size - (x2 - x2_clip)
    ky2 = kernel_size - (y2 - y2_clip)

    # Añadir el kernel al heatmap
    heatmap[y1_clip:y2_clip, x1_clip:x2_clip] += kernel[ky1:ky2, kx1:kx2] * 10.0

    heatmap_normalized = np.clip(heatmap / np.max(heatmap)*1.3, 0, 1)
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)

    # Aplicar colormap personalizado
    colormap_flat = colormap.reshape(256, 3)
    heatmap_color = colormap_flat[heatmap_uint8]
    
    # Mezclar heatmap_color con base_display
    alpha_channel = np.clip(heatmap_normalized * 0.6, 0, 1)[:, :, np.newaxis]  # Shape (height, width, 1)
    
    return heatmap_color, alpha_channel
    