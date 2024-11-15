import os
import h5py
import numpy as np
import torch
import random
import sys
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from dataset import MPIIFaceGazeDataset

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import Model

# tensorboard --logdir=logs/

def main(test_percentage, val_percentage, batch_size, num_epochs, learning_rate, weight_decay, data_path, num_subjects):
    seed = 30
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with h5py.File(os.path.join(data_path, 'data.h5'), 'r') as h5_file:
        person_ids = [fname.decode('utf-8').split('/')[0] for fname in h5_file['file_name_base']]
        max_person_idx = max(int(pid[1:]) for pid in person_ids)
        print(f"Máximo índice de persona en el dataset: {max_person_idx}")
        if max_person_idx >= num_subjects:
            print(f"Ajustando 'num_subjects' de {num_subjects} a {max_person_idx + 1}")
            num_subjects = max_person_idx + 1

    # Crear datasets y dataloaders 
    train_dataset = MPIIFaceGazeDataset(
        data_path=data_path, split='train',
        test_percentage=test_percentage, val_percentage=val_percentage
    )
    val_dataset = MPIIFaceGazeDataset(
        data_path=data_path, split='val',
        test_percentage=test_percentage, val_percentage=val_percentage
    )
    test_dataset = MPIIFaceGazeDataset(
        data_path=data_path, split='test',
        test_percentage=test_percentage, val_percentage=val_percentage
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Instanciar el modelo y definir características
    model = Model(num_subjects=num_subjects, learning_rate=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=True, mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath='./saved_models/gaze_estimation/',
        filename='gaze_model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    logger = TensorBoardLogger(
        save_dir='logs/',
        name='gaze_estimation',
        version=datetime.now().strftime("%Y%m%d-%H%M%S") 
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        default_root_dir='./saved_models',
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stopping],  
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Mejor modelo guardado en: {best_model_path}")
    model = Model.load_from_checkpoint(best_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer.test(model, test_dataloaders=test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de estimación de mirada')
    parser.add_argument('--test_percentage', '-tp', type=float, default=0.07, help='Porcentaje de días por persona a reservar para pruebas. Ej: 0.2')
    parser.add_argument('--val_percentage', '-vp', type=float, default=0.07, help='Porcentaje de días por persona a reservar para validación. Ej: 0.1')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Tamaño del lote para entrenamiento')
    parser.add_argument('--num_epochs', '-e', type=int, default=50, help='Número de épocas para entrenar')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Tasa de aprendizaje para el optimizador')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Decaimiento de peso para el optimizador')
    parser.add_argument('--data_path', '-dp', type=str, default='/home/fabian/Escritorio/MPIIFaceGaze_preprocessed/', help='Ruta del dataset preprocesado')
    parser.add_argument('--num_subjects', '-ns', type=int, default=15, help='Número de pacientes en el dataset')
    
    args = parser.parse_args()

    if not (0.0 <= args.test_percentage <= 1.0):
        raise ValueError("El parámetro --test_percentage debe estar entre 0 y 1.")
    if not (0.0 <= args.val_percentage <= 1.0):
        raise ValueError("El parámetro --val_percentage debe estar entre 0 y 1.")

    test_percentage = args.test_percentage
    val_percentage = args.val_percentage
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    data_path = args.data_path
    num_subjects = args.num_subjects

    main(test_percentage, val_percentage, batch_size, num_epochs, learning_rate, weight_decay, data_path, num_subjects)


