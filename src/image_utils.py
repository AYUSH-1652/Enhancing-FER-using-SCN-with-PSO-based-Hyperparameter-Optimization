import cv2
import numpy as np

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    
    noise = np.random.normal(mean, std, image_array.shape).astype(np.float32)
    image = image_array.astype(np.float32)
    
    noisy_img = image + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    
    return noisy_img.astype(np.uint8)


def flip_image(image_array):
    return cv2.flip(image_array, 1)


def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    gray_img_3d = np.stack([gray, gray, gray], axis=-1)
    return gray_img_3d