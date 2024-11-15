import os
import glob
import numpy as np
import cv2
import h5py
import pandas as pd
import scipy.io
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))) 

from utils.preprocessing import normalize_single_image, check_gaze_off_screen

def preprocess_dataset(input_path, output_path):
    """
    Preprocesa el dataset MPII Face Gaze y guarda las imágenes y datos procesados.

    Args:
        input_path (str): Ruta al dataset original.
        output_path (str): Ruta para guardar los datos procesados.
    """
    # Cargar el modelo de la cara
    face_model_path = os.path.join(input_path, '6 points-based face model.mat')
    face_model = scipy.io.loadmat(face_model_path)['model']

    data = defaultdict(list)

    # Procesar cada persona
    person_paths = sorted(glob.glob(os.path.join(input_path, 'p*')))
    for person_path in tqdm(person_paths, desc='Procesando personas'):
        person = os.path.basename(person_path)

        # Cargar datos de calibración de la cámara
        camera_mat_path = os.path.join(person_path, 'Calibration', 'Camera.mat')
        if not os.path.exists(camera_mat_path):
            print(f"Advertencia: No se encontró {camera_mat_path}. Saltando esta persona.")
            continue
        camera_data = scipy.io.loadmat(camera_mat_path)
        camera_matrix = camera_data['cameraMatrix']
        dist_coefficients = camera_data['distCoeffs']

        screen_size = scipy.io.loadmat(os.path.join(person_path, 'Calibration', 'screenSize.mat'))
        screen_width_pixel = screen_size["width_pixel"].item()
        screen_height_pixel = screen_size["height_pixel"].item()

        # Cargar anotaciones
        annotations_path = os.path.join(person_path, f'{person}.txt')
        if not os.path.exists(annotations_path):
            print(f"Advertencia: No se encontró {annotations_path}. Saltando esta persona.")
            continue

        annotations = pd.read_csv(annotations_path, sep='\s+', header=None, index_col=0)

        column_names = ['gaze_screen_x', 'gaze_screen_y']

        for i in range(1, 7):
            column_names.extend([f'landmark_{i}_x', f'landmark_{i}_y'])

        column_names.extend(['head_rot_x', 'head_rot_y', 'head_rot_z'])
        column_names.extend(['head_trans_x', 'head_trans_y', 'head_trans_z'])
        column_names.extend(['face_center_x', 'face_center_y', 'face_center_z'])
        column_names.extend(['gaze_target_x', 'gaze_target_y', 'gaze_target_z'])
        column_names.append('eye_used')

        # Verificar que la longitud de column_names coincide con el número de columnas
        if len(column_names) != annotations.shape[1]:
            print(f"Advertencia: Número de columnas no coincide para {person}. Esperado: {len(column_names)}, Actual: {annotations.shape[1]}")
            continue

        annotations.columns = column_names

        # Procesar cada día
        day_paths = sorted(glob.glob(os.path.join(person_path, 'day*')))
        for day_path in tqdm(day_paths, desc=f'Procesando días para {person}', leave=False):
            day = os.path.basename(day_path)
            image_files = sorted(glob.glob(os.path.join(day_path, '*.jpg')))

            for image_file in image_files:
                img = cv2.imread(image_file)
                if img is None:
                    print(f"Advertencia: No se pudo cargar la imagen {image_file}. Saltando.")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_name = os.path.basename(image_file)
                img_key = f"{day}/{img_name}"

                if img_key not in annotations.index:
                    print(f"Advertencia: {img_key} no está en las anotaciones. Saltando.")
                    continue

                annotation = annotations.loc[img_key]

                # Extraer los vectores de rotación y traslación
                head_rotation = annotation[['head_rot_x', 'head_rot_y', 'head_rot_z']].to_numpy().astype(float).flatten()
                head_translation = annotation[['head_trans_x', 'head_trans_y', 'head_trans_z']].to_numpy().astype(float).flatten()
                face_center = annotation[['face_center_x', 'face_center_y', 'face_center_z']].to_numpy().astype(float).reshape(3, 1)
                gaze_target_3d = annotation[['gaze_target_x', 'gaze_target_y', 'gaze_target_z']].to_numpy().astype(float).reshape(3, 1)
                head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
                face_landmarks_camera = head_rotation_matrix @ face_model + head_translation.reshape((3, 1))

                left_eye_indices = [2, 3] 
                right_eye_indices = [0, 1] 

                left_eye_center = 0.5 * (face_landmarks_camera[:, left_eye_indices[0]] + face_landmarks_camera[:, left_eye_indices[1]]).reshape((3, 1))
                right_eye_center = 0.5 * (face_landmarks_camera[:, right_eye_indices[0]] + face_landmarks_camera[:, right_eye_indices[1]]).reshape((3, 1))

                # Normalizar las imágenes
                img_warped_left_eye, _, _ = normalize_single_image(
                    img, head_rotation, None, left_eye_center, camera_matrix, is_eye=True
                )
                img_warped_right_eye, _, _ = normalize_single_image(
                    img, head_rotation, None, right_eye_center, camera_matrix, is_eye=True
                )
                img_warped_face, gaze_normalized, _ = normalize_single_image(
                    img, head_rotation, gaze_target_3d, face_center, camera_matrix, is_eye=False
                )

                # Calcular ángulos de mirada
                gaze_pitch = np.arcsin(-gaze_normalized[1])
                gaze_yaw = np.arctan2(-gaze_normalized[0], -gaze_normalized[2])

                # Guardar imágenes
                base_output_dir = os.path.join(output_path, person, day)
                os.makedirs(base_output_dir, exist_ok=True)
                base_filename = os.path.join(base_output_dir, img_name[:-4])
                cv2.imwrite(f'{base_filename}-left_eye.png', cv2.cvtColor(img_warped_left_eye, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'{base_filename}-right_eye.png', cv2.cvtColor(img_warped_right_eye, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'{base_filename}-full_face.png', cv2.cvtColor(img_warped_face, cv2.COLOR_RGB2BGR))

                # Recopilar datos
                data['file_name_base'].append(f'{person}/{day}/{img_name[:-4]}')
                data['gaze_pitch'].append(gaze_pitch.item())
                data['gaze_yaw'].append(gaze_yaw.item())
                data['gaze_location'].append([annotation['gaze_screen_x'], annotation['gaze_screen_y']])
                data['screen_size'].append([screen_width_pixel, screen_height_pixel])

    # Guardar datos en archivo HDF5
    with h5py.File(os.path.join(output_path, 'data.h5'), 'w') as h5_file:
        for key, value in data.items():
            if key == 'file_name_base':
                value = [s.encode('utf-8') for s in value]
                h5_file.create_dataset(key, data=value, compression='gzip')
            else:
                h5_file.create_dataset(key, data=np.array(value), compression='gzip')

    # Verificar imágenes donde la mirada no está en la pantalla
    check_gaze_off_screen(input_path, output_path)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocesar el dataset MPII Face Gaze')
    parser.add_argument('--input_path', "-i", type=str, required=True, help='Ruta al dataset original')
    parser.add_argument('--output_path', "-o", type=str, required=True, help='Ruta para guardar los datos procesados')
    args = parser.parse_args()
    preprocess_dataset(args.input_path, args.output_path)
