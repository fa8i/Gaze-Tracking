import pathlib
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import pandas as pd
from data_utils import show_point_on_screen

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from utils.functions_utils import get_monitor_dimensions
from utils.class_utils import WebcamSource

WINDOW_NAME = 'data collection'


def main(base_path: str, monitor_mm=None, monitor_pixels=None):
    pathlib.Path(f'{base_path}/').mkdir(parents=True, exist_ok=True)
    csv_path = f'{base_path}/data.csv'

    source = WebcamSource()
    next(source)  # start webcam

    if monitor_mm is None or monitor_pixels is None:
        monitor_mm, monitor_pixels = get_monitor_dimensions()
        if monitor_mm is None or monitor_pixels is None:
            raise ValueError('Please supply monitor dimensions manually as they could not be retrieved.')
    print(f'Found default monitor of size {monitor_mm[0]}x{monitor_mm[1]}mm and {monitor_pixels[0]}x{monitor_pixels[1]}px.')

    if pathlib.Path(csv_path).exists():
        existing_data = pd.read_csv(csv_path)
        last_index = existing_data.index[-1] + 1
    else:
        last_index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        collected_data = defaultdict(list)  # Reiniciar el diccionario en cada iteración

        file_name, center, time_till_capture = show_point_on_screen(WINDOW_NAME, base_path, monitor_pixels, source)
        if file_name is not None and time_till_capture is not None:
            collected_data['file_name'].append(file_name)
            collected_data['point_on_screen'].append(center)
            collected_data['time_till_capture'].append(time_till_capture)
            collected_data['monitor_mm'].append(monitor_mm)
            collected_data['monitor_pixels'].append(monitor_pixels)

            df = pd.DataFrame(collected_data)
            df.index = range(last_index, last_index + len(df))
            last_index += len(df)
            df.to_csv(csv_path, mode='a', header=not pathlib.Path(csv_path).exists(), index=True)

        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default='./data')
    parser.add_argument("--monitor_mm", type=str, help="milimetros de tu pantalla separado por comas")
    parser.add_argument("--monitor_pixels", type=str, help="pixeles de tu pantalla separado por comas")
    args = parser.parse_args()

    if args.monitor_mm is not None:
        args.monitor_mm = tuple(map(int, args.monitor_mm.split(',')))
    if args.monitor_pixels is not None:
        args.monitor_pixels = tuple(map(int, args.monitor_pixels.split(',')))

    main(args.base_path, args.monitor_mm, args.monitor_pixels)
