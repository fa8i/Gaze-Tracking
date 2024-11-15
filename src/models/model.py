import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(LightningModule):
    """
    Modelo para estimación de mirada basado en CNN con PyTorch Lightning.
    """

    def __init__(self, num_subjects=30, learning_rate=1e-4, weight_decay=0.0):
        """
        Args:
            num_subjects (int): Número total de sujetos en el dataset.
            learning_rate (float): Tasa de aprendizaje para el optimizador.
            weight_decay (float): Decaimiento de peso para el optimizador.
        """
        super(Model, self).__init__()
        self.save_hyperparameters()

        # Inicializar sesgos por sujeto
        self.subject_biases = nn.Parameter(torch.zeros(num_subjects, 2))  # [num_subjects, 2] para pitch y yaw

        vgg16_weights = models.VGG16_Weights.IMAGENET1K_V1

        self.cnn_face = nn.Sequential(
            models.vgg16(weights=vgg16_weights).features[:9],  # Primeras 9 capas de VGG16
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            models.vgg16(weights=vgg16_weights).features[:9],  # Primeras 9 capas de VGG16
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term

    def training_step(self, batch, batch_idx):
        left_eye = batch['left_eye_image']
        right_eye = batch['right_eye_image']
        full_face = batch['full_face_image']
        person_idx = batch['person_idx']
        gaze = batch['gaze']

        outputs = self(person_idx, full_face, right_eye, left_eye)
        loss = self.loss_fn(outputs, gaze)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        left_eye = batch['left_eye_image']
        right_eye = batch['right_eye_image']
        full_face = batch['full_face_image']
        person_idx = batch['person_idx']
        gaze = batch['gaze']

        outputs = self(person_idx, full_face, right_eye, left_eye)
        loss = self.loss_fn(outputs, gaze)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calcular error angular
        angular_err = self.angular_error(outputs, gaze)
        self.log('val_angular_error', angular_err, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        left_eye = batch['left_eye_image']
        right_eye = batch['right_eye_image']
        full_face = batch['full_face_image']
        person_idx = batch['person_idx']
        gaze = batch['gaze']

        outputs = self(person_idx, full_face, right_eye, left_eye)
        loss = self.loss_fn(outputs, gaze)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calcular error angular
        angular_err = self.angular_error(outputs, gaze)
        self.log('test_angular_error', angular_err, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=False),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def angular_error(self, predictions, labels):
        """Calcula el error angular entre las predicciones y las etiquetas.
        """
        # Convertir de ángulos a vectores unitarios 3D
        pred_vectors = self.angles_to_unit_vectors(predictions)
        label_vectors = self.angles_to_unit_vectors(labels)

        # Producto punto y cálculo del error angular
        cos_sim = torch.sum(pred_vectors * label_vectors, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angles = torch.acos(cos_sim) * (180 / torch.pi)
        return angles.mean()

    @staticmethod
    def angles_to_unit_vectors(angles):
        """Convierte ángulos de pitch y yaw a vectores unitarios 3D.
        """
        pitch = angles[:, 0]
        yaw = angles[:, 1]

        x = -torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)
        vectors = torch.stack([x, y, z], dim=1)
        return vectors