import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import imghdr



if __name__ == '__main__':
    data_dir = 'E:\comsat\Comsats\Semester7\Image processing\ProjectData'
    # image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    counter=0
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)

            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply median filtering
            original_image = cv2.medianBlur(original_image, 3)

            # Apply Canny edge detection
            edges_canny = cv2.Canny(original_image, 50, 150)

            # Apply morphological closing
            kernel = np.ones((7, 7), np.uint8)
            edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel)

            # Fill holes in the image
            edgesCanny_uint8 = edges_canny.astype(np.uint8)

            contours, _ = cv2.findContours(edgesCanny_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filled_image = np.zeros_like(edgesCanny_uint8)
            cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)


            path = "E:\comsat\Comsats\Semester7\Image processing\ProjectBlackandwhite"

            # Save the image using OpenCV
            cv2.imwrite(f"{path}/filename{counter}.png", filled_image)
            counter+=1
            # Display the result
            # plt.imshow(filled_image, cmap='gray')
            # plt.title('Image')
            # plt.show()

