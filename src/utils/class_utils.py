from dataclasses import dataclass
from typing import List
import collections
import time
import cv2
import numpy as np


@dataclass
class LandmarkPoint:
    """Clase que contiene las coordenadas x, y, z de cada landmark."""
    x: float
    y: float
    z: float

@dataclass
class EyeLandmarks:
    """Clase qie contiene listas de Landmarks de los ojos."""
    upper_eyelid: List[LandmarkPoint]
    lower_eyelid: List[LandmarkPoint]
    inner_side: List[LandmarkPoint]
    outer_side: List[LandmarkPoint]

@dataclass
class LandmarkSet:
    """Clase que contiene el set de landmarks."""
    all_landmarks: List[LandmarkPoint]
    left_eye: EyeLandmarks
    right_eye: EyeLandmarks


@dataclass
class ROIBox:
    """Clase que contiene la Region de interes (Region Of Interest Box)."""
    x: float
    y: float
    width: float
    height: float

class WebcamSource:
    """Clase auxiliar para OpenCV VideoCapture. Puede utilizarse como iterador.
    """
    def __init__(self, camera_id=0, width=1280, height=720, fps=30, buffer_size=1):
        self.__name = "WebcamSource"
        self.__capture = cv2.VideoCapture(camera_id)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.__capture.set(cv2.CAP_PROP_FPS, fps)
        self.__capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.buffer_size = buffer_size

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.fps_deque = collections.deque(maxlen=fps)

    def __iter__(self):
        if not self.__capture.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        """Get next frame from webcam or stop iteration when no frame can be grabbed from webcam
        """
        ret, frame = self.__capture.read()

        if not ret:
            raise StopIteration

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration

        return frame

    def clear_frame_buffer(self):
        for _ in range(self.buffer_size):
            self.__capture.read()

    def __del__(self):
        self.__capture.release()
        cv2.destroyAllWindows()

    def show(self, frame, only_print=False):
        self.new_frame_time = time.time()
        self.fps_deque.append(1 / (self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time

        if only_print:
            print(f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')
        else:
            cv2.imshow('show_frame', frame)
            cv2.setWindowTitle("show_frame", f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')

class Keyboard:
    """Clase auxiliar para mostrar teclado interactivo en pantalla"""
    def __init__(self, monitor_pixels, side_margin=100):
        self.monitor_width, self.monitor_height = monitor_pixels
        self.side_margin = side_margin  # Margen lateral en píxeles
        self.keys = self.create_keys()
        self.key_width = (self.monitor_width - 2 * self.side_margin) // max(len(row) for row in self.keys)
        self.key_height = self.monitor_height // 10  # Aumenta la altura de las teclas
        self.selected_key = None
        self.text = ""
        self.v_margin = 200

    def create_keys(self):
        # Define las filas del teclado QWERTY con la disposición solicitada
        rows = [
            list("1234567890"),
            list("QWERTYUIOP"),
            list("ASDFGHJKL")+["N'"],
            list("ZXCVBNM") + ["Backspace"],
            ["Espacio"]
        ]
        return rows

    def draw(self, frame):
        total_rows = len(self.keys)
        y_start = self.monitor_height - (self.key_height * total_rows) - self.v_margin 

        for row_idx, row in enumerate(self.keys):
            # Calcular el ancho total de la fila
            total_row_width = 0
            key_widths = []
            for key in row:
                if key == "Espacio":
                    key_width = self.key_width * 5
                elif key == "Backspace":
                    key_width = self.key_width * 2
                else:
                    key_width = self.key_width
                key_widths.append(key_width)
                total_row_width += key_width

            # Calcular el x_start para centrar la fila
            x_start = (self.monitor_width - total_row_width) // 2

            y_position = y_start + row_idx * self.key_height

            for key_idx, key in enumerate(row):
                current_key_width = key_widths[key_idx]

                top_left = (x_start, y_position)
                bottom_right = (top_left[0] + current_key_width - 10, top_left[1] + self.key_height - 10)  # 10 px de margen entre teclas
                cv2.rectangle(frame, top_left, bottom_right, (200, 200, 200), -1)
                cv2.rectangle(frame, top_left, bottom_right, (50, 50, 50), 2)

                # Dibujar el texto de la tecla
                font_scale = 0.7
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = key
                if key == "Espacio":
                    text = "SPACE"
                text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
                text_x = top_left[0] + (current_key_width - text_size[0]) // 2 - 5
                text_y = top_left[1] + (self.key_height + text_size[1]) // 2 - 5
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), 2)

                # Actualizar x_start para la siguiente tecla
                x_start += current_key_width

        # Mostrar el texto escrito encima del teclado
        text_y_position = y_start - 100  # 100 px por encima del teclado
        cv2.rectangle(frame, (self.side_margin, text_y_position - 40), 
                      (self.monitor_width - self.side_margin, text_y_position + 10), 
                      (255, 255, 255), -1)
        cv2.rectangle(frame, (self.side_margin, text_y_position - 40), 
                      (self.monitor_width - self.side_margin, text_y_position + 10), 
                      (50, 50, 50), 2)
        font_scale = 1.2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.text, font, font_scale, 2)[0]
        text_x = (self.monitor_width - text_size[0]) // 2
        text_y = text_y_position
        cv2.putText(frame, self.text, (text_x, text_y), font, font_scale, (0, 0, 0), 2)

    def get_key_at_point(self, point):
        x, y = point
        total_rows = len(self.keys)
        y_start = self.monitor_height - (self.key_height * total_rows) - self.v_margin 

        for row_idx, row in enumerate(self.keys):
            key_widths = []
            total_row_width = 0
            for key in row:
                if key == "Espacio":
                    key_width = self.key_width * 5
                elif key == "Backspace":
                    key_width = self.key_width * 2
                else:
                    key_width = self.key_width
                key_widths.append(key_width)
                total_row_width += key_width

            x_start = (self.monitor_width - total_row_width) // 2
            y_position = y_start + row_idx * self.key_height

            for key_idx, key in enumerate(row):
                current_key_width = key_widths[key_idx]

                top_left = (x_start, y_position)
                bottom_right = (top_left[0] + current_key_width - 10, top_left[1] + self.key_height - 10)
                if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                    return key

                x_start += current_key_width

        return None

    def add_character(self, key):
        if key in ["Espacio", "SPACE"]:
            self.text += " "
        elif key == "Backspace":
            self.text = self.text[:-1]
        else:
            self.text += key.lower()
