import streamlit as st
import numpy as np
from PIL import Image
import io

# Функция для выполнения SVD и восстановления изображения
def compress_image(image_array, num_singular_values):
    # Преобразование изображения в градации серого
    U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
    
    # Сжатие изображения, используя только 'num_singular_values'
    compressed_image = np.dot(U[:, :num_singular_values], np.dot(np.diag(S[:num_singular_values]), Vt[:num_singular_values, :]))
    
    return compressed_image

# Интерфейс Streamlit
st.title("Приложение для SVD")

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Выберете объект...", type="jpg")

if uploaded_file is not None:
    # Открытие изображения
    image = Image.open(uploaded_file)
    
    # Преобразование изображения в градации серого для упрощенной обработки
    image_gray = image.convert("L")
    
    # Преобразование изображения в массив numpy
    image_array = np.array(image_gray)
    
    # Выбор количества сингулярных чисел
    num_singular_values = st.slider('Выберете количество сингулярных значений', 1, min(image_array.shape), 50)
    
    # Отображение оригинального изображения
    st.image(image, caption='Оригинальный вид', use_column_width=True)
    
    # Сжатие изображения
    compressed_image_array = compress_image(image_array, num_singular_values)
    
    # Преобразование массива обратно в изображение
    compressed_image = Image.fromarray(np.uint8(compressed_image_array))
    
    # Отображение сжатого изображения
    st.image(compressed_image, caption='После сжатия', use_column_width=True)