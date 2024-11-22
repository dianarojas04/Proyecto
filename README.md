# Proyecto

## Integrantes
Juan Carlos Cañón Cárdenas 
Ibsen Anneth Sánchez Hernández
Diana Carolina Rojas Orjuela

## Descripción
Este proyecto utiliza técnicas de procesamiento de imágenes para diagnosticar glaucoma a partir de imágenes de fondo de ojo. El glaucoma es una enfermedad ocular grave que puede llevar a ceguera irreversible si no se detecta a tiempo. El objetivo principal es calcular la relación copa-disco (CDR), un parámetro clave para evaluar la presencia de glaucoma. Este trabajo se basa en el análisis de imágenes del dataset ORIGA-light.

## Requisitos
### Librerías necesarias
Python 3.11.9
OpenCV
NumPy
Matplotlib
Math
### Dataset
El proyecto emplea imágenes del dataset ORIGA-light, que contiene 650 imágenes de retina con anotaciones realizadas por expertos para investigación en glaucoma.

## Flujo de trabajo
### Carga de la imagen
Se carga la imagen en formato RGB y se divide en canales para un análisis detallado del canal verde y rojo.
### Procesamiento del canal verde 
Se aplica un filtro Sobel para detectar bordes.
Se realiza umbralización con el método de Otsu.
La máscara resultante es suavizada con operaciones morfológicas y filtros de mediana.
Finalmente, se rellena mediante interpolación para eliminar imperfecciones.
### Procesamiento del canal rojo
Se aplica un filtro gaussiano para suavizar el ruido.
Se calcula un histograma para identificar las intensidades dominantes y definir umbrales adecuados para segmentación.
### Segmentación de estructuras
Se detectan los contornos de la copa y el disco óptico.
Se calcula el área de estas estructuras y se encierra cada una en círculos mínimos.
### Cálculo del CDR
Se emplea la fórmula:
𝐶𝐷𝑅=Área de la Copa/Área del Disco
Valores de CDR ≥ 0.6 indican sospecha de glaucoma.
### Resultados visuales
Imágenes procesadas y máscaras generadas.
Histogramas y estructuras segmentadas destacadas en la imagen original.

## Ejecución del código
Coloque las imágenes de entrada en el directorio del proyecto y asegúrese de que los nombres coincidan con los utilizados en el código (158.jpg en este caso).
Instale las dependencias con: pip install opencv-python matplotlib numpy
Ejecute el código principal: python Codigo.py
Revise los resultados en las imágenes generadas y las salidas impresas en consola.