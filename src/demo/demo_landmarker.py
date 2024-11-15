import sys
import os
import cv2
import argparse
import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from landmarker.mediapipe_landmarker import MediaPipeLandmarker
from landmarker.blendshape_logger import BlendshapeLogger
from utils.functions_utils import calculate_bounding_box, extract_angles
from utils.class_utils import WebcamSource
import utils.constant_utils as cte

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
tesselation = cte.FACEMESH_TESSELATION 


def parse_arguments():
    parser = argparse.ArgumentParser(description="Detección de Landmark con MediaPipe.")
    parser.add_argument('--blendshapes', '-bs', action='store_true', default=False,
                        help='Habilita el registro de blendshapes.')
    parser.add_argument('--matrix', '-m', action='store_true', default=False,
                        help='Habilita la visualización de ángulos y ejes de referencia.')
    parser.add_argument('--bounding_boxes', '-bb', action='store_true', default=False,
                        help='Habilita el dibujo de bounding boxes.')
    parser.add_argument('--facemesh', '-fm', action='store_true', default=False,
                        help='Habilita el dibujo de la máscara facial.')
    return parser.parse_args()


def draw_axis(result, matrix, frame):
    pitch, yaw, roll, t = extract_angles(matrix)
    text_pitch = f"Pitch: {np.degrees(pitch):.2f}"
    text_yaw = f"Yaw: {np.degrees(yaw):.2f}"
    text_roll = f"Roll: {np.degrees(roll):.2f}"
    text_translation = f"Translation: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"

    cv2.putText(frame, text_pitch, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
    cv2.putText(frame, text_yaw, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(frame, text_roll, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.putText(frame, text_translation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    nose_landmark = result.all_landmarks[1]
    origin_x = int(nose_landmark.x)
    origin_y = int(nose_landmark.y)
    origin = (origin_x, origin_y)

    R = matrix[:3, :3]

    x_axis = np.array([1, 0, 0])  # Eje X hacia la izquierda
    y_axis = np.array([0, -1, 0])  # Eje Y hacia arriba
    z_axis = np.array([0, 0, 1])   # Eje Z hacia adelante (hacia la cámara)

    x_rot = R @ x_axis
    y_rot = R @ y_axis
    z_rot = R @ z_axis

    x_end = (int(origin_x + x_rot[0] * 150), int(origin_y - x_rot[1] * 150))
    y_end = (int(origin_x - y_rot[0] * 150), int(origin_y + y_rot[1] * 150))
    z_end = (int(origin_x + z_rot[0] * 250), int(origin_y - z_rot[1] * 250))

    cv2.line(frame, origin, x_end, BLUE, 2)
    cv2.line(frame, origin, y_end, GREEN, 2)
    cv2.line(frame, origin, z_end, RED, 2)


def draw_facemesh(result, frame):
    for edge in tesselation:
        start_idx, end_idx = edge
        if start_idx < len(result.all_landmarks) and end_idx < len(result.all_landmarks):
            p1 = result.all_landmarks[start_idx]
            p2 = result.all_landmarks[end_idx]
            x1, y1 = int(p1.x), int(p1.y)
            x2, y2 = int(p2.x), int(p2.y)
            cv2.line(frame, (x1, y1), (x2, y2), GREEN, 1)
    for landmark in result.all_landmarks:
        x, y = int(landmark.x), int(landmark.y)
        cv2.circle(frame, (x, y), 1, (255,0,255), -1)


def main():
    args = parse_arguments()
    csv_filename = 'blendshapes.csv'
    blendshapes = args.blendshapes
    transformation_matrixes = args.matrix
    frame_interval = 5

    face_margins = (-0.1, 0.0)
    eye_margins = (0.25, 0.5)

    landmarker = MediaPipeLandmarker(
        tracker=True,
        blendshapes=blendshapes,
        transformation_matrixes=transformation_matrixes
    )

    webcam = WebcamSource(camera_id=0, width=1280, height=720, fps=30)

    logger = None
    if blendshapes:
        logger = BlendshapeLogger(filename=csv_filename, frame_interval=frame_interval)
        print(f"BlendshapeLogger inicializado. Guardando cada {frame_interval} frames en {csv_filename}.")

    for frame in webcam:
        result, blends, matrix = landmarker.detect(frame)

        if result and args.bounding_boxes:
            face_bbox = calculate_bounding_box(result.all_landmarks, face_margins, aspect_ratio=1.0)

            right_eye_landmarks = (
                result.right_eye.upper_eyelid + result.right_eye.lower_eyelid +
                result.right_eye.inner_side + result.right_eye.outer_side)
            right_eye_bbox = calculate_bounding_box(right_eye_landmarks, eye_margins, aspect_ratio=2/1)

            left_eye_landmarks = (
                result.left_eye.upper_eyelid + result.left_eye.lower_eyelid +
                result.left_eye.inner_side + result.left_eye.outer_side)
            left_eye_bbox = calculate_bounding_box(left_eye_landmarks, eye_margins, aspect_ratio=2/1)

            # Cara
            cv2.rectangle(frame, (face_bbox.x, face_bbox.y), (face_bbox.x + face_bbox.width, face_bbox.y + face_bbox.height), BLUE, 2)
            # Ojo derecho
            cv2.rectangle(frame, (right_eye_bbox.x, right_eye_bbox.y),
                            (right_eye_bbox.x + right_eye_bbox.width, right_eye_bbox.y + right_eye_bbox.height), GREEN, 2)
            # Ojo izquierdo
            cv2.rectangle(frame, (left_eye_bbox.x, left_eye_bbox.y),
                            (left_eye_bbox.x + left_eye_bbox.width, left_eye_bbox.y + left_eye_bbox.height), GREEN, 2)

        if result and args.facemesh:
            draw_facemesh(result, frame)

        if blends and logger:
            logger.log(blends)

        if args.matrix and matrix is not None:
            draw_axis(result, matrix, frame)

        webcam.show(frame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
