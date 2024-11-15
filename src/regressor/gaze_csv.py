import sys
import os
import pandas as pd
from argparse import ArgumentParser

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import Model
from utils import constant_utils as cte
from utils.preprocessing import normalize_single_image
from utils.functions_utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen

face_model_all = cte.FACE_MODEL
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # Ojo derecho, ojo izquierdo, boca
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def extract_screen_size(input_path):
    df = pd.read_csv(input_path)
    monitor_pixels = df['monitor_pixels'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
    monitor_mm = df['monitor_mm'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
    if (monitor_pixels.nunique() == 1).all() and (monitor_mm.nunique() == 1).all():
        return monitor_mm.iloc[0].values, monitor_pixels.iloc[0].values
    else:
        raise ValueError("El tamaño del monitor en mm o en píxeles no es consistente entre todas las filas del archivo.")

def process_images(calibration_matrix_path: str, images_folder: str, csv_file: str, model=None):
    camera_matrix, dist_coefficients = get_camera_matrix(calibration_matrix_path)
    data = pd.read_csv(csv_file)
    
    monitor_mm, monitor_pixels = extract_screen_size(csv_file)
    print(f'Monitor detectado de tamaño {monitor_mm[0]}x{monitor_mm[1]}mm y {monitor_pixels[0]}x{monitor_pixels[1]}px.')

    # Configurar el plano de la pantalla (asumimos que está en el origen y es perpendicular al eje Z)
    plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
    plane_w = plane[0:3]
    plane_b = plane[3]

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    gaze_pitch_list = []
    gaze_yaw_list = []
    point_on_screen_x_list = []
    point_on_screen_y_list = []
    rvec_list = []
    tvec_list = []

    # Loop a través de cada imagen en el CSV
    for idx, row in data.iterrows():
        file_name = row['file_name']
        image_path = os.path.join(images_folder, file_name)
        if not os.path.exists(image_path):
            print(f'Imagen {image_path} no encontrada.')
            gaze_pitch_list.append(np.nan)
            gaze_yaw_list.append(np.nan)
            point_on_screen_x_list.append(np.nan)
            point_on_screen_y_list.append(np.nan)
            rvec_list.append([np.nan, np.nan, np.nan])
            tvec_list.append([np.nan, np.nan, np.nan])
            continue

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f'No se pudo leer la imagen {image_path}.')
            gaze_pitch_list.append(np.nan)
            gaze_yaw_list.append(np.nan)
            point_on_screen_x_list.append(np.nan)
            point_on_screen_y_list.append(np.nan)
            rvec_list.append([np.nan, np.nan, np.nan])
            tvec_list.append([np.nan, np.nan, np.nan])
            continue

        height, width, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            # Estimación de la pose de la cabeza
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

            success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, flags=cv2.SOLVEPNP_EPNP)
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

            # Preprocesamiento de datos
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, image_rgb.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)

            # Transformar imágenes
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(device) 
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(device)

            # Predicción
            with torch.no_grad():
                output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).cpu().numpy()
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            gaze_pitch, gaze_yaw = gaze_vector[0], gaze_vector[1]

            # Calcular el punto en pantalla
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

            gaze_pitch_list.append(gaze_pitch)
            gaze_yaw_list.append(gaze_yaw)
            point_on_screen_x_list.append(point_on_screen[0])
            point_on_screen_y_list.append(point_on_screen[1])
            rvec_list.append(rvec.flatten())
            tvec_list.append(tvec.flatten())

            print(f'Procesada imagen {file_name}: gaze_pitch={gaze_pitch}, gaze_yaw={gaze_yaw}, point_on_screen={point_on_screen}')
        else:
            print(f'No se detectaron landmarks faciales en la imagen {file_name}.')
            gaze_pitch_list.append(np.nan)
            gaze_yaw_list.append(np.nan)
            point_on_screen_x_list.append(np.nan)
            point_on_screen_y_list.append(np.nan)
            rvec_list.append([np.nan, np.nan, np.nan])
            tvec_list.append([np.nan, np.nan, np.nan])

    data['gaze_pitch'] = gaze_pitch_list
    data['gaze_yaw'] = gaze_yaw_list
    data['point_on_screen_x'] = point_on_screen_x_list
    data['point_on_screen_y'] = point_on_screen_y_list
    data['rvec'] = rvec_list
    data['tvec'] = tvec_list

    # Guardar nuevo csv
    script_directory = os.path.dirname(os.path.abspath(__file__)) 
    csv_file_name = os.path.basename(csv_file)
    output_csv_file = os.path.join(script_directory, csv_file_name.replace('.csv', '_with_gaze.csv')) 

    try:
        data.to_csv(output_csv_file, index=False)
        print(f'Se guardó el CSV actualizado con los datos adicionales en {output_csv_file}')
    except Exception as e:
        print(f'Error al guardar el CSV: {e}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--calibration_matrix_path", type=str, default='../data_collection/calibration_matrix.yaml')
    parser.add_argument("--model_path", type=str, default='../models/best_cnn_model.ckpt')
    parser.add_argument("--images_folder", type=str, default='../data_collection/data')
    parser.add_argument("--csv_file", type=str, default='../data_collection/data/data.csv')
    args = parser.parse_args()

    model = Model.load_from_checkpoint(args.model_path).to(device)
    model.eval()

    process_images(args.calibration_matrix_path, args.images_folder, args.csv_file, model)
