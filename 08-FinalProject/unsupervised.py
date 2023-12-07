import os
import cv2
import numpy as np
from skimage import color
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar imágenes
def cargar_imagenes(ruta):
    imagenes = []
    for archivo in os.listdir(ruta):
        if archivo.endswith(".jpg"):
            imagen = cv2.imread(os.path.join(ruta, archivo))
            imagenes.append(imagen)
    return imagenes

# Ruta de la carpeta con las imágenes
ruta_imagenes = "bottle/plastic"

# Cargar imágenes
imagenes = cargar_imagenes(ruta_imagenes)

# Preprocesamiento de imágenes
def preprocesar_imagen(imagen):
    # Convertir a espacio de color Lab
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    # Escalar los valores para mejorar la convergencia del algoritmo K-Means
    lab = lab / 255.0
    return lab

# Segmentación usando el algoritmo SLIC
def segmentar_imagen(imagen):
    segmentacion = slic(imagen, n_segments=100, compactness=10, sigma=1)
    return segmentacion

# Extracción de características para cada segmento
def extraer_caracteristicas(imagen, segmentacion):
    caracteristicas = []
    for segmento_id in np.unique(segmentacion):
        mascara = (segmentacion == segmento_id).astype(int)
        media_color = np.mean(imagen * np.expand_dims(mascara, axis=-1), axis=(0, 1))
        caracteristicas.append(np.concatenate([media_color]))
    return caracteristicas

# Agrupamiento con K-Means
def clusterizar_caracteristicas(caracteristicas, num_clusters):
    modelo = KMeans(n_clusters=num_clusters, random_state=42)
    modelo.fit(caracteristicas)
    return modelo.labels_

# Preprocesamiento, segmentación, y extracción de características para todas las imágenes
imagenes_preprocesadas = [preprocesar_imagen(img) for img in imagenes]
segmentaciones = [segmentar_imagen(img) for img in imagenes_preprocesadas]
caracteristicas = [extraer_caracteristicas(img, seg) for img, seg in zip(imagenes_preprocesadas, segmentaciones)]
caracteristicas = np.concatenate(caracteristicas)

# Clustering con K-Means
num_clusters = 2  # Puedes ajustar este valor
etiquetas = clusterizar_caracteristicas(caracteristicas, num_clusters)

# Imprimir resultados
for i, etiqueta in enumerate(etiquetas):
    print(f"Imagen {i + 1}: {'Botella' if etiqueta == 1 else 'No botella'}")

# Visualizar la segmentación y los clusters
def visualizar_resultados(imagen, segmentacion, etiquetas):
    # Visualizar la segmentación
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(imagen)
    axes[0].set_title('Imagen Original')

    axes[1].imshow(segmentacion, cmap='viridis')
    axes[1].set_title('Segmentación')

    # Visualizar los clusters
    cluster_image = np.zeros_like(segmentacion, dtype=np.uint8)
    for i, label in enumerate(np.unique(etiquetas)):
        cluster_image[segmentacion == i] = label

    axes[2].imshow(cluster_image, cmap='viridis')
    axes[2].set_title('Clusters')

    plt.show()

# Visualizar resultados para cada imagen
for i, (imagen, seg, etiqueta) in enumerate(zip(imagenes_preprocesadas, segmentaciones, etiquetas)):
    print(f"Imagen {i + 1}: {'Botella' if etiqueta == 1 else 'No botella'}")
    visualizar_resultados(imagen, seg, etiquetas)