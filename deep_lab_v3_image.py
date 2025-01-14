import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np

# Especificar la configuración del ImageSegmenter
options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path="./Models/deeplab_v3.tflite"),
    output_category_mask=True,
    running_mode=vision.RunningMode.IMAGE)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Categorías
categories ={0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tv'}
# Asignar colores únicos a las categorías
category_colors = np.random.randint(0, 255, size=(len(categories), 3), dtype="uint8")

# Leer la imagen de entrada
image = cv2.imread("./Inputs/Imagen_8.png")

# Convertir la imagen a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Obtener los resultados del segmentador
segmentation_result = segmenter.segment(image_rgb)

# Convertir la máscara de categorías en un array de Numpy
category_mask = segmentation_result.category_mask
category_mask_np = category_mask.numpy_view()

# Transformar el category mask a 3 canales
category_mask_bgr = cv2.cvtColor(category_mask_np, cv2.COLOR_GRAY2BGR)

# Pintar cada segmento con su color correspondiente
for category_id in np.unique(category_mask_np):
    color = category_colors[category_id]
    category_mask_bgr[np.where(category_mask_np == category_id)] = color

# Aplicar transparencia
alpha = 0.5
final_image = cv2.addWeighted(image, 1 - alpha, category_mask_bgr, alpha, 0)

# VISUALIZAR CATEGORÍAS
black_image = np.zeros((430, 200, 3), dtype="uint8")
y_offset = 20
font_scale = 0.6
line_thickness = 2

for category_id, name in categories.items():
    if category_id in np.unique(category_mask_np):
        color = tuple(map(int, category_colors[category_id]))
    else:
        color = (128, 128, 128)
    cv2.putText(black_image,
                f"{category_id}: {name}",
                (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_thickness,
                cv2.LINE_AA)
    y_offset += 20

# Visualizar imagen
image_scale = cv2.resize(image, (540, 540))
final_image_scale = cv2.resize(final_image, (540, 540))
cv2.imshow("Image", image_scale)
cv2.imshow("Final image", final_image_scale)
cv2.imshow("Black image", black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()