from typing import Tuple
import cv2
import os
import glob
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm


def get_matrices(camera_matrix: np.ndarray, distance_norm: int, center_point: np.ndarray, focal_norm: int, head_rotation_matrix: np.ndarray, image_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula las matrices de rotación, escalado y transformación.

    Args:
        camera_matrix (np.ndarray): Matriz de la cámara.
        distance_norm (int): Distancia normalizada desde la cámara.
        center_point (np.ndarray): Punto central en la imagen.
        focal_norm (int): Focal normalizada.
        head_rotation_matrix (np.ndarray): Rotación de la cabeza.
        image_output_size (Tuple[int, int]): Tamaño de la imagen de salida.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Matrices de rotación, escalado y transformación.
    """
    dist_to_center = np.linalg.norm(center_point)  # Distancia real al punto central
    scaling_factor = distance_norm / dist_to_center  # Factor de escala basado en la distancia

    norm_cam_matrix = np.array([
        [focal_norm, 0, image_output_size[0] / 2],
        [0, focal_norm, image_output_size[1] / 2],
        [0, 0, 1]
    ])

    scale_matrix = np.diag([1, 1, scaling_factor])

    forward_vec = (center_point / dist_to_center).reshape(3)
    down_vec = np.cross(forward_vec, head_rotation_matrix[:, 0])
    down_vec /= np.linalg.norm(down_vec)
    right_vec = np.cross(down_vec, forward_vec)
    right_vec /= np.linalg.norm(right_vec)

    rot_matrix = np.vstack([right_vec, down_vec, forward_vec])
    transform_matrix = norm_cam_matrix @ scale_matrix @ rot_matrix @ np.linalg.inv(camera_matrix)

    return rot_matrix, scale_matrix, transform_matrix


def equalize_hist_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """
    Ecualiza el histograma de una imagen RGB.

    Args:
        rgb_img (np.ndarray): Imagen RGB.

    Returns:
        np.ndarray: Imagen RGB con el histograma ecualizado.
    """
    # Convertimos la imagen de RGB a YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    # Ecualizamos solo el canal de luminancia
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    # Volvemos a convertir la imagen a RGB
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)


def normalize_single_image(image: np.ndarray, head_rotation, gaze_target: np.ndarray, center_point: np.ndarray, camera_matrix: np.ndarray, is_eye: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza una imagen individual, creando una imagen normalizada de ojo o rostro según el valor de `is_eye`.

    Args:
        image (np.ndarray): Imagen original.
        head_rotation (np.ndarray): Rotación de la cabeza.
        gaze_target (np.ndarray): Punto objetivo de la mirada.
        center_point (np.ndarray): Punto 3D de referencia en el rostro.
        camera_matrix (np.ndarray): Matriz de la cámara.
        is_eye (bool): Indica si se está procesando una imagen del ojo.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Imagen normalizada, vector de mirada normalizado y matriz de rotación.
    """
    # Parámetros de cámara normalizados
    focal_length_norm = 960
    distance_norm = 500 if is_eye else 1600
    img_output_size = (96, 64) if is_eye else (96, 96)

    if gaze_target is not None:
        gaze_target = gaze_target.reshape((3, 1))

    head_rot_matrix, _ = cv2.Rodrigues(head_rotation)
    rot_matrix, scale_matrix, transform_matrix = get_matrices(camera_matrix, distance_norm, center_point, focal_length_norm, head_rot_matrix, img_output_size)

    # Aplicamos la transformación a la imagen
    warped_img = cv2.warpPerspective(image, transform_matrix, img_output_size)
    # Ecualizamos el histograma de la imagen
    warped_img = equalize_hist_rgb(warped_img)

    if gaze_target is not None:
        # Normalizamos el vector de mirada
        normalized_gaze = gaze_target - center_point
        normalized_gaze = rot_matrix @ normalized_gaze
        normalized_gaze /= np.linalg.norm(normalized_gaze)
    else:
        normalized_gaze = np.zeros(3)

    return warped_img, normalized_gaze.reshape(3), rot_matrix

def check_gaze_off_screen(input_path, output_path):
    """
    Verifica las imágenes donde la mirada no está en la pantalla y guarda sus nombres de archivo.

    Args:
        input_path (str): Ruta al dataset original.
        output_path (str): Ruta para guardar el archivo CSV con errores.
    """
    data = {'file_name': [], 'on_screen_gaze_position': [], 'monitor_pixels': []}

    person_paths = sorted(glob.glob(os.path.join(input_path, 'p*')))
    for person_path in tqdm(person_paths, desc='Verificando mirada fuera de pantalla'):
        person = os.path.basename(person_path)

        screen_size = scipy.io.loadmat(os.path.join(person_path, 'Calibration', 'screenSize.mat'))
        screen_width_pixel = screen_size["width_pixel"].item()
        screen_height_pixel = screen_size["height_pixel"].item()

        annotations = pd.read_csv(os.path.join(person_path, f'{person}.txt'), sep=' ', header=None)

        day_paths = sorted(glob.glob(os.path.join(person_path, 'day*')))
        df_idx = 0

        for day_path in day_paths:
            day = os.path.basename(day_path)
            image_files = sorted(glob.glob(os.path.join(day_path, '*.jpg')))

            for image_file in image_files:
                row = annotations.iloc[df_idx]
                gaze_x, gaze_y = row[1:3].to_numpy().astype(int)

                if not (0 <= gaze_x <= screen_width_pixel and 0 <= gaze_y <= screen_height_pixel):
                    file_name = os.path.join(person, day, os.path.basename(image_file))

                    data['file_name'].append(file_name)
                    data['on_screen_gaze_position'].append([gaze_x, gaze_y])
                    data['monitor_pixels'].append([screen_width_pixel, screen_height_pixel])

                df_idx += 1

    df_errors = pd.DataFrame(data)
    df_errors.to_csv(os.path.join(output_path, 'not_on_screen.csv'), index=False)