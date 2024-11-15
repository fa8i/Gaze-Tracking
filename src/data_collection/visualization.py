import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import sys
import yaml

from argparse import ArgumentParser
from typing import Tuple
from matplotlib import pyplot as plt

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from utils.functions_utils import get_camera_matrix, get_face_landmarks_in_ccs
from utils import constant_utils as cte

face_model_all = cte.FACE_MODEL
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])  # fijar ejes
face_model_all *= 10

landmarks_ids = cte.LANDMARKS_IDS
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])


def setup_figure() -> Tuple:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-400, 400)
    ax.set_ylim(-100, 700)
    ax.set_zlim(-10, 800 - 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax


def plot_screen(ax, screen_width_mm, screen_height_mm, screen_height_mm_offset) -> None:
    ax.plot(0, 0, 0, linestyle="", marker="o", color='#1f77b4', label='webcam')

    screen_x = [-screen_width_mm / 2, screen_width_mm / 2]
    screen_y = [screen_height_mm_offset, screen_height_mm + screen_height_mm_offset]
    ax.plot(
        [screen_x[0], screen_x[1], screen_x[1], screen_x[0], screen_x[0]],
        [screen_y[0], screen_y[0], screen_y[1], screen_y[1], screen_y[0]],
        [0, 0, 0, 0, 0],
        color='#ff7f0e',
        label='screen'
    )


def plot_target_on_screen(ax, point_on_screen_px, monitor_mm, monitor_pixels, screen_height_mm_offset):
    screen_width_ratio = monitor_mm[0] / monitor_pixels[0]
    screen_height_ratio = monitor_mm[1] / monitor_pixels[1]

    point_on_screen_mm = (monitor_mm[0] / 2 - point_on_screen_px[0] * screen_width_ratio, point_on_screen_px[1] * screen_height_ratio + screen_height_mm_offset)
    ax.plot(point_on_screen_mm[0], point_on_screen_mm[1], 0, linestyle="", marker="X", color='#9467bd', label='target on screen')
    return point_on_screen_mm[0], point_on_screen_mm[1], 0


def plot_face_landmarks(ax, face_model_all_transformed):
    ax.plot(face_model_all_transformed[0, :], face_model_all_transformed[1, :], face_model_all_transformed[2, :], linestyle="", marker="o", color='#7f7f7f', markersize=1, label='face landmarks')


def plot_eye_to_target_on_screen_line(ax, face_model_all_transformed, point_on_screen_3d):
    eye_center = (face_model_all_transformed[:, 33] + face_model_all_transformed[:, 133]) / 2
    ax.plot([point_on_screen_3d[0], eye_center[0]], [point_on_screen_3d[1], eye_center[1]], [point_on_screen_3d[2], eye_center[2]], color='#2ca02c', label='right eye gaze vector')

    eye_center = (face_model_all_transformed[:, 263] + face_model_all_transformed[:, 362]) / 2
    ax.plot([point_on_screen_3d[0], eye_center[0]], [point_on_screen_3d[1], eye_center[1]], [point_on_screen_3d[2], eye_center[2]], color='#d62728', label='left eye gaze vector')


def main(data_path: str, cam_matrix_path: str, output_path: str, screen_height_mm_offset: int = 10):
    os.makedirs(output_path, exist_ok=True)
    fix_qt_cv_mismatch()

    camera_matrix, dist_coefficients = get_camera_matrix(cam_matrix_path)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    df = pd.read_csv(f'{data_path}/data.csv')
    for idx, row in df.iterrows():
        monitor_mm = tuple(map(int, row['monitor_mm'][1:-1].split(',')))
        monitor_pixels = tuple(map(int, row['monitor_pixels'][1:-1].split(',')))
        point_on_screen_px = tuple(map(int, row['point_on_screen'][1:-1].split(',')))

        fig, ax = setup_figure()
        plot_screen(ax, monitor_mm[0], monitor_mm[1], screen_height_mm_offset)
        point_on_screen_3d = plot_target_on_screen(ax, point_on_screen_px, monitor_mm, monitor_pixels, screen_height_mm_offset)

        frame = cv2.imread(f'{data_path}/{row["file_name"]}')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # mark the image as not writeable to pass by reference
        results = face_mesh.process(frame_rgb)

        if results is None or results.multi_face_landmarks is None:
            print('Mediapipe could not detect a face.')
            continue

        _, face_model_all_transformed,  = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, frame.shape, results)
        plot_face_landmarks(ax, face_model_all_transformed)
        plot_eye_to_target_on_screen_line(ax, face_model_all_transformed, point_on_screen_3d)

        ax.view_init(-70, -90)
        plt.legend()
        plt.tight_layout()
        
        output_file_path = f'{output_path}/3d_plot_{idx}.png'
        plt.savefig(output_file_path)
        plt.show()

        print(f"Imagen {row['file_name']} guardada en: {output_file_path}")


def fix_qt_cv_mismatch():
    import os
    for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, default='./data', help="Ruta donde se almacenan las capturas y el csv")
    parser.add_argument("--cam_matrix_path", "-cm", type=str, default='.', help="Ruta donde se encuentra el archivo callibration_matrix.yaml")
    parser.add_argument("--output_path", "-o", type=str, default="./images", help="Ruta donde se guardarán las imágenes")
    args = parser.parse_args()

    main(args.data_path, args.cam_matrix_path, args.output_path)
