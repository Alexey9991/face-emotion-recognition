import cv2
import os

def convert_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def remove_noise(image):
    denoised_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised_image

input_path = "FER-2013/test/angry"
output_path = "image_preprocessed/test/angry"


if not os.path.exists(output_path):
    os.makedirs(output_path)


for root, dirs, files in os.walk(input_path):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

          
            gray_image = convert_to_gray(image)

           
            denoised_image = remove_noise(gray_image)

            
            relative_path = os.path.relpath(root, input_path)
            output_folder = os.path.join(output_path, relative_path)
            output_filename = os.path.join(output_folder, filename)

            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)


            cv2.imwrite(output_filename, denoised_image)
