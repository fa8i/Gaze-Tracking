import cv2
import numpy as np
import os
import glob
import yaml

def create_calibration_folder(folder_path):
    """Crea la carpeta de calibración si no existe."""
    os.makedirs(folder_path, exist_ok=True)

def get_next_image_number(folder_path, prefix='image', extension='.jpg'):
    """Obtiene el siguiente número disponible para nombrar una imagen.

    Args:
        folder_path (str): Ruta a la carpeta de calibración.
        prefix (str, optional): Prefijo de los archivos de imagen. Por defecto es 'image'.
        extension (str, optional): Extensión de los archivos de imagen. Por defecto es '.jpg'.

    Returns:
        int: El siguiente número disponible para la imagen.
    """
    existing_images = glob.glob(os.path.join(folder_path, f"{prefix}*{extension}"))
    numbers = [
        int(os.path.splitext(os.path.basename(img))[0].replace(prefix, ''))
        for img in existing_images
        if os.path.splitext(os.path.basename(img))[0].replace(prefix, '').isdigit()
    ]
    return max(numbers) + 1 if numbers else 1

def capture_images(folder_path):
    """Captura imágenes desde la cámara web y las guarda en la carpeta de calibración.

    Args:
        folder_path (str): Ruta a la carpeta de calibración.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cv2.namedWindow('Captura de Imágenes para Calibración', cv2.WINDOW_NORMAL)
    print("Presiona 'Espacio' o 'Enter' para capturar una imagen.")
    print("Presiona 'q' o 'Esc' para finalizar la captura y proceder a la calibración.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        cv2.imshow('Captura de Imágenes para Calibración', frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(' '), 13]:
            img_number = get_next_image_number(folder_path)
            img_name = f"image{img_number}.jpg"
            cv2.imwrite(os.path.join(folder_path, img_name), frame)
            print(f"Imagen guardada: {img_name}")

        elif key in [ord('q'), 27]:
            print("Finalizando la captura de imágenes.")
            break

    cap.release()
    cv2.destroyAllWindows()

def calibrate_camera(folder_path, chessboard_size=(9,6), save_file='calibration_matrix.yaml', show_corners=True):
    """Calibra la cámara utilizando imágenes capturadas de un chess-grid.

    Args:
        folder_path (str): Ruta a la carpeta de calibración.
        chessboard_size (tuple, optional): Tamaño de ches-sgrid.
        save_file (str, optional): Nombre del archivo para guardar los datos de calibración.
        show_corners (bool, optional): Indica si se deben mostrar las esquinas detectadas. 
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(folder_path, 'image*.jpg'))

    if not images:
        print("No se encontraron imágenes para la calibración.")
        return

    if show_corners:
        cv2.namedWindow('Esquinas Encontradas', cv2.WINDOW_NORMAL)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"No se pudo leer la imagen: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            if show_corners:
                img_con_esquinas = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners2, ret)
                cv2.imshow('Esquinas Encontradas', img_con_esquinas)
                cv2.waitKey(100)
                print(f"Esquinas detectadas en: {fname}")
        else:
            print(f"Esquinas no encontradas en la imagen: {fname}")

    if show_corners:
        cv2.destroyWindow('Esquinas Encontradas')

    if len(objpoints) < 15:
        print(f"Advertencia: Solo se encontraron {len(objpoints)} imágenes válidas para la calibración. Se recomienda usar al menos 15 imágenes.")

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\nMatriz de la cámara (Matriz intrínseca):")
    print(camera_matrix)
    print("\nCoeficientes de distorsión:")
    print(dist_coeffs)
    print(f"\nError RMS de calibración: {ret}")

    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeff': dist_coeffs.tolist(),
        'rms': ret
    }
    with open(save_file, 'w') as file:
        yaml.dump(calibration_data, file)
    
    print(f"\nParámetros de calibración guardados en: {save_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_folder = os.path.join(script_dir, 'calibration')
    save_path = os.path.join(script_dir, 'calibration_matrix.yaml')
    create_calibration_folder(calibration_folder)
    capture_images(calibration_folder)
    calibrate_camera(calibration_folder, chessboard_size=(9,6), save_file=save_path, show_corners=True)

if __name__ == "__main__":
    main()
