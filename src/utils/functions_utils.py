from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import List, Optional, Tuple
from .class_utils import ROIBox, LandmarkPoint


def calculate_bounding_box(
    landmarks: List[LandmarkPoint],
    margins: Tuple[float, float] = (0.0, 0.0),
    aspect_ratio: Optional[float] = None,  # Relación de aspecto deseada (ancho / alto)
    image: Optional[np.ndarray] = None     # Imagen como array de NumPy para obtener dimensiones
) -> ROIBox:
    """Calcula el bounding box que engloba una lista de LandmarkPoint, ajustando con márgenes
    y opcionalmente ajustando la relación de aspecto deseada sin comprobar los límites de la imagen.

    Args:
        landmarks (List[LandmarkPoint]): Lista de puntos de landmarks.
        margins (Tuple[float, float]): Márgenes como porcentaje adicional para el ancho y alto.
        aspect_ratio (Optional[float]): Relación de aspecto deseada (ancho / alto). Si es None, no se ajusta.

    Returns:
        ROIBox: Bounding box ajustado.
    """
    if not landmarks:
        raise ValueError("La lista de landmarks está vacía.")

    points = np.array([[point.x, point.y] for point in landmarks])

    # Calcular los límites mínimos y máximos en x y y
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width, height = x_max - x_min, y_max - y_min

    # Aplicar márgenes
    width *= (1 + 2 * margins[0])
    height *= (1 + 2 * margins[1])

    # Calcular el centro del bbox original y recalcular x_min y y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, y_min = x_center - width / 2, y_center - height / 2

    if aspect_ratio is not None and height != 0:
        current_ratio = width / height
        if current_ratio < aspect_ratio:    # Aumentar ancho
            width = aspect_ratio * height
            x_min = x_center - width / 2
        elif current_ratio > aspect_ratio:  # Aumentar alto para cumplir la relación de aspecto
            height = width / aspect_ratio
            y_min = y_center - height / 2
        
    x_min=int(round(x_min))
    y_min=int(round(y_min))
    width=int(round(width))
    height=int(round(height))
    
    return ROIBox(x_min, y_min, width, height)


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """Graficar los blendshgapes en una gráfica de barras.

    Args:
        face_blendshapes (dict): Diccionario que contiene el nombre de los blendshapes y su valor normalizado.
    """
    face_blendshapes_names = face_blendshapes.keys()
    face_blendshapes_scores = face_blendshapes.values()
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def extract_angles(transformation_matrix):
    """Transforma una matriz de transformación 4x4 en ángulos de Euler (pitch, yaw, roll) y vector de traslación.
    
    Args:
        transformation_matrix (numpy.ndarray): Matriz de transformación 4x4.
        
    Returns:
        tuple:
            pitch (float): Ángulo de pitch.
            yaw (float): Ángulo de yaw.
            roll (float): Ángulo de roll.
            t (numpy.ndarray): Vector de traslación 3x1.
    """
    # Extraer matriz de rotacion (3x3) y vector de traslacion (3x1).
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # Calcular angulos de Euler
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0

    # pitch = np.degrees(pitch)
    # yaw = np.degrees(yaw)
    # roll = np.degrees(roll)
    
    return pitch, yaw, roll, t


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """Obtiene las dimensiones del monitor desde Gdk. Requiere importacion de pgi (puede causar problemas de instalacion en Windows)

    Fuente: https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]] | Tuple[None, None]:
            Tupla con el ancho y alto del monitor en mm y píxeles, o (None, None) si no se pudo obtener.
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository # type: ignore

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk # type: ignore

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def get_camera_matrix(calibration_matrix_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga la `camera_matrix` y los coeficientes de distorsión desde `{cam_matrix_path}/calibration_matrix.yaml`.

    Args:
        cam_matrix_path (str): Ruta de los datos.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matriz intrínseca de la cámara y coeficientes de distorsión.
    """
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients


def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """Ajusta el `face_model` a los `face_landmarks` usando `solvePnP`.

    Args:
        camera_matrix (np.ndarray): Matriz intrínseca de la cámara.
        dist_coefficients (np.ndarray): Coeficientes de distorsión.
        shape (tuple): Dimensiones de la imagen (altura, anchura, canales).
        results: Salida de MediaPipe FaceMesh.
        face_model (np.ndarray): Modelo 3D de la cara.
        face_model_all (np.ndarray): Modelo completo de la cara con todos los puntos de referencia.
        landmarks_ids (list): IDs de los puntos de referencia faciales a usar.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Posiciones 3D de los puntos de referencia faciales en el sistema de coordenadas de la cámara.
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks


def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """Convierte un vector de mirada 2D (pitch y yaw) a un vector de dirección 3D.

    Args:
        gaze (np.ndarray): Vector de mirada con pitch y yaw.

    Returns:
        np.ndarray: Vector de dirección 3D.
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])


def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """Calcula el punto de intersección entre el rayo de mirada y el plano que representa la pantalla.

    Args:
        support_vector (np.ndarray): Vector de soporte del rayo de mirada.
        direction_vector (np.ndarray): Vector de dirección del rayo de mirada.
        plane_normal (np.ndarray): Vector normal del plano.
        plane_d (np.ndarray): Parámetro d del plano.

    Returns:
        np.ndarray: Punto 3D en la pantalla donde la mirada intersecta.
    """
    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    return np.linalg.solve(matrix, bias).reshape(3)


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """Calcula la ecuación de un plano en el espacio 3D usando la matriz de rotación y traslación.

    Args:
        rmat (np.ndarray): Matriz de rotación.
        tmat (np.ndarray): Matriz de traslación.

    Returns:
        np.ndarray: Coeficientes (a, b, c, d) de la ecuación del plano: ax + by + cz = d.
    """

    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
    assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """Convierte un punto en milímetros a coordenadas en píxeles en la pantalla.

    Args:
        monitor_mm (Tuple[float, float]): Dimensiones del monitor en milímetros.
        monitor_pixels (Tuple[float, float]): Dimensiones del monitor en píxeles.
        result (np.ndarray): Punto calculado en milímetros en la pantalla.

    Returns:
        Tuple[int, int]: Coordenadas del punto en píxeles en la pantalla.
    """
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])

    return tuple(np.asarray([result_x, result_y]).round().astype(int))
