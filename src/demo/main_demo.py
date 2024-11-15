import os
import sys
import collections
import joblib
import time
from argparse import ArgumentParser

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import Model
from landmarker.mediapipe_landmarker import MediaPipeLandmarker
from utils import constant_utils as cte
from utils.preprocessing import normalize_single_image
from utils.functions_utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from utils.demo_utils import create_custom_colormap, predict_point_on_screen, calc_heatmap
from utils.class_utils import WebcamSource, Keyboard


face_model_all = cte.FACE_MODEL
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = cte.LANDMARKS_IDS
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

WINDOW_NAME = 'laser pointer preview'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

laser_color = (0, 0, 255)  # Color del puntero 
laser_alpha = 0.8   # Transparencia del puntero
browDown_threshold = 0.09 

def main(calibration_matrix_path: str, monitor_mm=None, monitor_pixels=None, model=None, visualize_preprocessing=False, smoothing=False, heatmap_enabled=False, keyboard_enabled=False):
    # setup webcam
    source = WebcamSource(width=1280, height=720, fps=30, buffer_size=10)
    camera_matrix, dist_coefficients = get_camera_matrix(calibration_matrix_path)

    # setup monitor
    if monitor_mm is None or monitor_pixels is None:
        monitor_mm, monitor_pixels = get_monitor_dimensions()
        if monitor_mm is None or monitor_pixels is None:
            raise ValueError('Por favor, introduzca las dimensiones del monitor manualmente')
    print(f'Medidas encontradas del monitor: {monitor_mm[0]}x{monitor_mm[1]}mm and {monitor_pixels[0]}x{monitor_pixels[1]}px.')

    plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))  # Posible calibración de la posición de la pantalla // corregida modelo de regresión
    plane_w = plane[0:3]
    plane_b = plane[3]

    fps_deque = collections.deque(maxlen=60)  # Medir FPS
    prev_frame_time = 0

    smoothing_buffer = collections.deque(maxlen=3)
    rvec_buffer = collections.deque(maxlen=3)
    tvec_buffer = collections.deque(maxlen=3)
    gaze_vector_buffer = collections.deque(maxlen=10)
    rvec, tvec = None, None
    gaze_points = collections.deque(maxlen=64)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ema_point = None  # Inicializar variable de suavizado
    alpha = 0.15  # Factor de suavizado 

    if heatmap_enabled:
        base_display_initial = cv2.imread('images/web.png')
        if base_display_initial is None:
            raise ValueError('No se encontró la imagen web.png en el directorio actual.')
        base_display_initial = cv2.resize(base_display_initial, (monitor_pixels[0], monitor_pixels[1]))
        heatmap = np.zeros((monitor_pixels[1], monitor_pixels[0]), dtype=np.float32)
        colormap = create_custom_colormap()
    else:
        base_display_initial = np.ones((monitor_pixels[1], monitor_pixels[0], 3), dtype=np.uint8) * 255  # Fondo blanco

    if keyboard_enabled:
        keyboard = Keyboard(monitor_pixels, side_margin=100)  # Ajusta el margen lateral según sea necesario
        landmarker = MediaPipeLandmarker(blendshapes=True)

    key_selection_cooldown = 0  # Evitar múltiples selecciones rápidas

    for frame_idx, frame in enumerate(source):
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        base_display = base_display_initial.copy()   # Reiniciar base_display en cada frame

        if results.multi_face_landmarks:
            # head pose estimation
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])
            smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(smoothing_buffer).mean(axis=0)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Primera estimación
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Segunda estimación para mayor accuracy

            rvec_buffer.append(rvec)
            rvec = np.asarray(rvec_buffer).mean(axis=0)
            tvec_buffer.append(tvec)
            tvec = np.asarray(tvec_buffer).mean(axis=0)

            # data preprocessing
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, frame.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # Centro ojo izquierdo
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # Centro ojo derecho
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))  # Centro cara (media 3D de los 7 landmarks usados para resolver PnP)

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)

            if visualize_preprocessing:
                cv2.imshow('img_warped_left_eye', cv2.cvtColor(img_warped_left_eye, cv2.COLOR_RGB2BGR))
                cv2.imshow('img_warped_right_eye', cv2.cvtColor(img_warped_right_eye, cv2.COLOR_RGB2BGR))
                cv2.imshow('img_warped_face', cv2.cvtColor(img_warped_face, cv2.COLOR_RGB2BGR))

            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(device)   # idxtensor 1x1 64bit
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(device)

            output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()  # prediccion gaze pitch, yaw
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            gaze_vector_buffer.append(gaze_vector)
            gaze_vector = np.asarray(gaze_vector_buffer).mean(axis=0)

            # gaze vector to screen
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)  # Calculo de la interseccion vector y plano pantalla
            point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)  # Conversion a píxeles
            point_on_screen_corrected = predict_point_on_screen(    # Estimación del punto real
                gaze_vector,
                point_on_screen,  
                rvec.flatten(),
                tvec.flatten(),
                scaler,
                regressor,
                monitor_pixels
            )

            if heatmap_enabled:
                gaze_x, gaze_y = point_on_screen_corrected
                heatmap_color, alpha_channel = calc_heatmap(gaze_x, gaze_y, monitor_pixels, heatmap, colormap)
                overlay = heatmap_color.astype(np.float32) * alpha_channel
                base_display_float = base_display.astype(np.float32) * (1 - alpha_channel)
                result_image = cv2.add(base_display_float, overlay).astype(np.uint8)
            else:
                overlay = base_display.copy()

                if smoothing:
                    if ema_point is None:
                        ema_point = np.array(point_on_screen_corrected, dtype=np.float32)
                    else:
                        ema_point = alpha * np.array(point_on_screen_corrected, dtype=np.float32) + (1 - alpha) * ema_point

                    ema_point_int = tuple(ema_point.astype(int))
                    cv2.circle(overlay, ema_point_int, radius=10, color=laser_color, thickness=-1)
                else:
                    gaze_points.appendleft(point_on_screen_corrected)
                    for idx in range(1, len(gaze_points)):
                        thickness = round((len(gaze_points) - idx) / len(gaze_points) * 5) + 1
                        cv2.line(overlay, gaze_points[idx - 1], gaze_points[idx], laser_color, thickness)

                # Aplicar transparencia: mezcla el overlay con el base_display
                cv2.addWeighted(overlay, laser_alpha, base_display, 1 - laser_alpha, 0, base_display)

                result_image = base_display.copy()

            if keyboard_enabled:
                keyboard.draw(result_image)

                if smoothing:
                    pointer_pos = tuple(ema_point.astype(int))
                else:
                    pointer_pos = tuple(point_on_screen_corrected)

                cv2.circle(result_image, pointer_pos, radius=15, color=laser_color, thickness=2)
                _, blendshapes, _ = landmarker.detect(frame)
                browDown_value = 0.0

                if blendshapes:
                    browDownLeft = blendshapes.get('browDownLeft')
                    browDownRight = blendshapes.get('browDownRight')

                    if browDownLeft is not None and browDownRight is not None:
                        browDown_value = np.mean([browDownLeft, browDownRight], dtype=np.float64)

                if browDown_value > browDown_threshold and key_selection_cooldown == 0:
                    selected_key = keyboard.get_key_at_point(pointer_pos)
                    if selected_key:
                        keyboard.add_character(selected_key)
                        print(f"Tecla seleccionada: {selected_key}")
                        key_selection_cooldown = 10

                if key_selection_cooldown > 0:
                    key_selection_cooldown -= 1

            cv2.imshow(WINDOW_NAME, result_image)

        new_frame_time = time.time()
        fps_deque.append(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        if frame_idx % 60 == 0:
            print(f'FPS: {np.mean(fps_deque):5.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--calibration_matrix_path", type=str, default='../data_collection/calibration_matrix.yaml')
    parser.add_argument("--model_path", type=str, default='../models/best_cnn_model.ckpt')
    parser.add_argument("--monitor_mm", type=str, default='344,215')
    parser.add_argument("--monitor_pixels", type=str, default='1920,1200')
    parser.add_argument("--visualize_preprocessing", '-vp', action='store_true')
    parser.add_argument("--smoothing", "-s", action='store_true', help="Aplicar suavizado exponencial al punto de mirada")
    parser.add_argument("--heatmap", "-hm", action='store_true', help="Usar imagen de fondo y mostrar heatmap de mirada")
    parser.add_argument("--keyboard", "-k", action='store_true', help="Mostrar teclado QWERTY en pantalla")
    args = parser.parse_args()

    if args.monitor_mm is not None:
        args.monitor_mm = tuple(map(int, args.monitor_mm.split(',')))
    if args.monitor_pixels is not None:
        args.monitor_pixels = tuple(map(int, args.monitor_pixels.split(',')))

    model = Model.load_from_checkpoint(args.model_path).to(device)
    model.eval()
    regressor = joblib.load('../models/CatBoost_model.joblib')
    scaler = joblib.load('../models/robust_scaler.joblib')

    main(args.calibration_matrix_path, args.monitor_mm, args.monitor_pixels, model, args.visualize_preprocessing, args.smoothing, args.heatmap, args.keyboard)
