from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import ica_module
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/independent-component-analysis")
def calculo(samples: int):
    output_file_1 = 'ica_signs.png'
    output_file_2 = 'ica_dispersion.png'
    # Generar datos de ejemplo
    np.random.seed(0)
    n_samples = samples
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Señal 1: onda sinusoidal
    s2 = np.sign(np.sin(3 * time))  # Señal 2: onda cuadrada
    s3 = np.random.normal(size=n_samples)  # Señal 3: ruido gaussiano

    S = np.c_[s1, s2, s3]
    S /= S.std(axis=0)  # Normalizar

    # Mezclar las señales
    A = np.array([[1, 1, 1], [0.5, 2, 0.5], [1.5, 1, 2]])  # Matriz de mezcla
    X = np.dot(S, A.T)  # Mezcla de señales

    # Convertir a tensor de PyTorch
    X_tensor = torch.from_numpy(X).float()

    # Aplicar ICA
    num_componentes = 3
    S_estimated, W = ica_module.ica(X_tensor, num_componentes)
    S_estimated = S_estimated.detach().numpy()

    # Graficar las señales originales
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title("Señales Originales")
    for i in range(S.shape[1]):
        plt.plot(time, S[:, i], label=f"Señal {i+1}")
    plt.legend()

    # Graficar las señales mezcladas
    plt.subplot(3, 1, 2)
    plt.title("Señales Mezcladas")
    for i in range(X.shape[1]):
        plt.plot(time, X[:, i], label=f"Mezcla {i+1}")
    plt.legend()

    # Graficar las señales separadas
    plt.subplot(3, 1, 3)
    plt.title("Señales Separadas por ICA")
    for i in range(S_estimated.shape[1]):
        plt.plot(time, S_estimated[:, i], label=f"Componente {i+1}")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(output_file_1)

    # Graficar diagramas de dispersión de las señales separadas
    plt.figure(figsize=(12, 4))
    plt.title("Diagramas de Dispersión de las Señales Separadas")
    for i in range(S_estimated.shape[1]):
        plt.scatter(time, S_estimated[:, i], label=f"Componente {i+1}", alpha=0.5)
    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    #plt.show()

    plt.savefig(output_file_2)
    plt.close()
    
    j1 = {
        "Grafica de señales": output_file_1,
        "Grafica de dispersión": output_file_2
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/independent-component-analysis-graphs")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
