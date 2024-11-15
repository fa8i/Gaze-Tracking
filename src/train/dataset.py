import os
import h5py
import torch
from torch.utils.data import Dataset
from skimage import io
from collections import defaultdict
import numpy as np
import random
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2

class MPIIFaceGazeDataset(Dataset):
    def __init__(self, data_path, split='train', test_percentage=0.2, val_percentage=0.1):
        """
        Args:
            data_path (str): Ruta al directorio de datos preprocesados.
            split (str): 'train', 'val' o 'test' indicando qué conjunto usar.
            test_percentage (float): Porcentaje de días por persona a reservar para pruebas.
            val_percentage (float): Porcentaje de días por persona a reservar para validación.
        """
        assert split in ['train', 'val', 'test'], "El parámetro 'split' debe ser 'train', 'val' o 'test'."
        self.data_path = data_path
        self.split = split
        self.test_percentage = test_percentage
        self.val_percentage = val_percentage

        # Cargar datos desde data.h5
        with h5py.File(os.path.join(data_path, 'data.h5'), 'r') as h5_file:
            self.file_names = [fname.decode('utf-8') for fname in h5_file['file_name_base']]
            self.gaze_pitch = h5_file['gaze_pitch'][:]
            self.gaze_yaw = h5_file['gaze_yaw'][:]
            self.person_ids = [fname.split('/')[0] for fname in self.file_names]
            self.day_ids = [fname.split('/')[1] for fname in self.file_names]

        # Construir un diccionario para mapear persona y día a índices de datos
        self.person_day_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (person_id, day_id) in enumerate(zip(self.person_ids, self.day_ids)):
            self.person_day_to_indices[person_id][day_id].append(idx)

        self.to_tensor = ToTensorV2()

        # Dividir datos en entrenamiento, validación y prueba según el porcentaje por persona
        self.indices = []
        for person_id, days_dict in self.person_day_to_indices.items():
            days = list(days_dict.keys())
            random.shuffle(days)  # Mezclar los días aleatoriamente

            total_days = len(days)
            num_test_days = int(round(total_days * self.test_percentage))
            remaining_days = total_days - num_test_days
            num_val_days = int(round(remaining_days * self.val_percentage))

            # Asignar días a cada conjunto
            test_days = days[:num_test_days]
            val_days = days[num_test_days:num_test_days + num_val_days]
            train_days = days[num_test_days + num_val_days:]

            print(f"Persona: {person_id}, Total días: {total_days}, Días de prueba: {len(test_days)}, Días de validación: {len(val_days)}, Días de entrenamiento: {len(train_days)}")

            if self.split == 'train':
                selected_days = train_days
            elif self.split == 'val':
                selected_days = val_days
            else:  # 'test'
                selected_days = test_days

            for day in selected_days:
                self.indices.extend(days_dict[day])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        base_name = self.file_names[actual_idx]
        left_eye_path = os.path.join(self.data_path, base_name + '-left_eye.png')
        right_eye_path = os.path.join(self.data_path, base_name + '-right_eye.png')
        face_path = os.path.join(self.data_path, base_name + '-full_face.png')

        # Cargar imágenes
        left_eye_img = io.imread(left_eye_path)
        right_eye_img = io.imread(right_eye_path)
        face_img = io.imread(face_path)

        if left_eye_img.ndim == 2:
            raise ValueError(f"Imagen en blanco y negro detectada en {left_eye_path}. Todas las imágenes deben ser en color.")
        if right_eye_img.ndim == 2:
            raise ValueError(f"Imagen en blanco y negro detectada en {right_eye_path}. Todas las imágenes deben ser en color.")
        if face_img.ndim == 2:
            raise ValueError(f"Imagen en blanco y negro detectada en {face_path}. Todas las imágenes deben ser en color.")

        # Aplicar normalización y conversión a tensor
        left_eye_img = self.to_tensor(image=left_eye_img.astype(np.float32))['image']
        right_eye_img = self.to_tensor(image=right_eye_img.astype(np.float32))['image']
        face_img = self.to_tensor(image=face_img.astype(np.float32))['image']

        # Obtener etiquetas
        gaze_pitch = self.gaze_pitch[actual_idx]
        gaze_yaw = self.gaze_yaw[actual_idx]
        gaze_vector = torch.tensor([gaze_pitch, gaze_yaw], dtype=torch.float32)

        # Obtener índice de persona (para los sesgos de sujeto)
        person_id = self.person_ids[actual_idx]
        person_idx = int(person_id[1:])  
        return {
            'left_eye_image': left_eye_img,
            'right_eye_image': right_eye_img,
            'full_face_image': face_img,
            'gaze': gaze_vector,
            'person_idx': torch.tensor(person_idx, dtype=torch.long)
        }
