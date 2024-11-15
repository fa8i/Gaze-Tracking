import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import argparse

warnings.filterwarnings('ignore')

def extract_screen_size(input_path):
    df = pd.read_csv(input_path)
    monitor_pixels = df['monitor_pixels'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
    if (monitor_pixels.nunique() == 1).all():
        return monitor_pixels.iloc[0].values
    else:
        raise ValueError("El tamaño del monitor en píxeles no es consistente entre todas las filas del archivo.")

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos de regresión con diferentes algoritmos y guardar los resultados.')
    parser.add_argument('--input_path', '-i', type=str, default=os.path.join(os.path.dirname(__file__), 'data_with_gaze.csv'), help='Ruta del archivo CSV de entrada.')
    parser.add_argument('--output_path', '-o', type=str, default=os.path.dirname(__file__), help='Ruta donde se guardarán los modelos y gráficos.')
    parser.add_argument('--save_graphs', '-sg', action='store_true', help='Guardar los gráficos generados en la ruta de salida.')
    parser.add_argument('--random_state', '-rs', type=int, default=69, help='Valor del random state para asegurar la reproducibilidad.')
    args = parser.parse_args()

    r_s = args.random_state

    screen_width, screen_height = extract_screen_size(args.input_path)

    dataset_path = args.input_path
    df = pd.read_csv(dataset_path)

    df.drop(['Unnamed: 0', 'file_name', 'time_till_capture', 'monitor_mm', 'monitor_pixels'], axis=1, inplace=True)
    df[['point_on_screen_x_real', 'point_on_screen_y_real']] = df['point_on_screen'].str.extract(r'\((\d+),\s*(\d+)\)').astype(float)
    df.drop('point_on_screen', axis=1, inplace=True)

    def parse_vector_column(df, col):
        return pd.DataFrame(df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')).tolist(),
                            columns=[f"{col}_{axis}" for axis in ['x', 'y', 'z']])

    rvec_components = parse_vector_column(df, 'rvec')
    tvec_components = parse_vector_column(df, 'tvec')
    df = pd.concat([df.drop(['rvec', 'tvec'], axis=1), rvec_components, tvec_components], axis=1)

    features = ['gaze_pitch', 'gaze_yaw', 'point_on_screen_x', 'point_on_screen_y',
                'rvec_x', 'rvec_y', 'rvec_z', 'tvec_x', 'tvec_y', 'tvec_z']
    target = ['point_on_screen_x_real', 'point_on_screen_y_real']

    X = df[features].copy()
    y = df[target].copy()

    # Normalizar
    X['point_on_screen_x'] = X['point_on_screen_x'] / screen_width
    X['point_on_screen_y'] = X['point_on_screen_y'] / screen_height
    X[['tvec_x', 'tvec_y', 'tvec_z']] = X[['tvec_x', 'tvec_y', 'tvec_z']] / 1000

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[['gaze_pitch', 'gaze_yaw', 'rvec_x', 'rvec_y', 'rvec_z']])

    X_scaled_df = pd.DataFrame(X_scaled, columns=['gaze_pitch', 'gaze_yaw', 'rvec_x', 'rvec_y', 'rvec_z'])

    # Agregar las demás características ya normalizadas
    X_scaled_df['point_on_screen_x'] = X['point_on_screen_x']
    X_scaled_df['point_on_screen_y'] = X['point_on_screen_y']
    X_scaled_df['tvec_x'] = X['tvec_x']
    X_scaled_df['tvec_y'] = X['tvec_y']
    X_scaled_df['tvec_z'] = X['tvec_z']

    # Normalizar las variables objetivo
    y_normalized = y.copy()
    y_normalized['point_on_screen_x_real'] = y_normalized['point_on_screen_x_real'] / screen_width
    y_normalized['point_on_screen_y_real'] = y_normalized['point_on_screen_y_real'] / screen_height

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_normalized, test_size=0.2, random_state=r_s)

    models = {
        'Random Forest': RandomForestRegressor(random_state=r_s),
        'Extra Trees': ExtraTreesRegressor(random_state=r_s),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(random_state=r_s)),
        'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(random_state=r_s)),
        'CatBoost': MultiOutputRegressor(CatBoostRegressor(verbose=0, random_state=r_s))
    }

    # Entrenar modelos y calcular métricas
    results = {}
    for name, model in models.items():
        print(f"Entrenando el modelo: {name}")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        y_pred[:, 0] = y_pred[:, 0] * screen_width  # Desnormalizar X
        y_pred[:, 1] = y_pred[:, 1] * screen_height  # Desnormalizar Y

        # Revertir la normalización de y_test
        y_test_reverted = y_test.copy()
        y_test_reverted['point_on_screen_x_real'] = y_test_reverted['point_on_screen_x_real'] * screen_width
        y_test_reverted['point_on_screen_y_real'] = y_test_reverted['point_on_screen_y_real'] * screen_height

        # Calcular métricas para cada objetivo
        rmse_x = root_mean_squared_error(y_test_reverted['point_on_screen_x_real'], y_pred[:, 0])
        r2_x = r2_score(y_test_reverted['point_on_screen_x_real'], y_pred[:, 0])
        
        rmse_y = root_mean_squared_error(y_test_reverted['point_on_screen_y_real'], y_pred[:, 1])
        r2_y = r2_score(y_test_reverted['point_on_screen_y_real'], y_pred[:, 1])
        
        rmse_avg = (rmse_x + rmse_y) / 2
        r2_avg = (r2_x + r2_y) / 2
        
        # Guardar resultados
        results[name] = {
            'Model': model,
            'RMSE_X': rmse_x,
            'RMSE_Y': rmse_y,
            'RMSE_Avg': rmse_avg,
            'R2_X': r2_x,
            'R2_Y': r2_y,
            'R2_Avg': r2_avg
        }
        
        print(f"{name}: RMSE_X = {rmse_x:.4f}, RMSE_Y = {rmse_y:.4f}, RMSE_Avg = {rmse_avg:.4f}")
        print(f"{name}: R2_X = {r2_x:.4f}, R2_Y = {r2_y:.4f}, R2_Avg = {r2_avg:.4f}\n")

    # Seleccionar el mejor modelo basado en RMSE promedio
    best_model_name = min(results, key=lambda x: results[x]['RMSE_Avg'])
    best_model = results[best_model_name]['Model']

    print(f"Mejor modelo: {best_model_name} con RMSE Promedio = {results[best_model_name]['RMSE_Avg']:.4f}")

    # Guardar el scaler y el mejor modelo
    output_path = args.output_path
    joblib.dump(scaler, os.path.join(output_path, 'robust_scaler.joblib'))
    joblib.dump(best_model, os.path.join(output_path, f'{best_model_name}_model.joblib'))

    # Comparar predicciones y valores reales
    y_pred_df = pd.DataFrame(y_pred, columns=['x_pred', 'y_pred'])
    comparison_df = pd.concat([y_test_reverted.reset_index(drop=True), y_pred_df], axis=1)

    if args.save_graphs:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='point_on_screen_x_real', y='x_pred', data=comparison_df, alpha=0.6)
        plt.plot([0, screen_width], [0, screen_width], color='red', linestyle='--')  # Línea de referencia
        plt.xlabel('Valor Real X')
        plt.ylabel('Predicción X')
        plt.title('Comparación de X real vs. predicha')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x='point_on_screen_y_real', y='y_pred', data=comparison_df, alpha=0.6)
        plt.plot([0, screen_height], [0, screen_height], color='red', linestyle='--')  # Línea de referencia
        plt.xlabel('Valor Real Y')
        plt.ylabel('Predicción Y')
        plt.title('Comparación de Y real vs. predicha')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'comparacion_predicciones.png'), facecolor='white')
        plt.show()

        # Crear gráfica comparativa con R2 en la leyenda
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))

        for name, model_info in results.items():
            y_pred_x = model_info['Model'].predict(X_test)[:, 0] * screen_width  # Desnormalizar las predicciones para X
            r2_x = model_info['R2_X']
            axs[0].plot(y_pred_x, label=f'{name} - R2 X: {r2_x:.2f}')
        axs[0].plot(y_test_reverted['point_on_screen_x_real'].values, color='black', linestyle='--', label='Datos Reales')
        axs[0].set_title('Comparación de point_on_screen_x_real (Escala Original)')
        axs[0].set_xlabel('Índice de prueba')
        axs[0].set_ylabel('Coordenada X en píxeles')
        axs[0].legend()

        for name, model_info in results.items():
            y_pred_y = model_info['Model'].predict(X_test)[:, 1] * screen_height  # Desnormalizar las predicciones para Y
            r2_y = model_info['R2_Y']
            axs[1].plot(y_pred_y, label=f'{name} - R2 Y: {r2_y:.2f}')
        axs[1].plot(y_test_reverted['point_on_screen_y_real'].values, color='black', linestyle='--', label='Datos Reales')
        axs[1].set_title('Comparación de point_on_screen_y_real (Escala Original)')
        axs[1].set_xlabel('Índice de prueba')
        axs[1].set_ylabel('Coordenada Y en píxeles')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'comparacion_modelos_R2.png'), facecolor='white')
        plt.show()

if __name__ == "__main__":
    main()
