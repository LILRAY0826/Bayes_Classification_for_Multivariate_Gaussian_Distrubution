import cv2
import numpy as np
import model
import matplotlib.pyplot as plt

# Load voc img
voc_img = cv2.imread("label_to_voc_dataset/voc_dataset/SegmentationClassPNG/full_duck.png")
org_img = cv2.imread("label_to_voc_dataset/label_dataset/full_duck.jpeg")

'''''''''
# Image cutting for testing
voc_img = voc_img[1500:2000, 1000:3000]
org_img = org_img[1500:2000, 1000:3000]
'''''''''

duck_pixels, non_duck_pixels = model.find_duck_and_non_duck_pixels(voc_img)
duck_rgb, non_duck_rgb = model.find_duck_and_non_duck_rgb(org_img, duck_pixels, non_duck_pixels)

print("Training, please wait a second.")
# Get the feature vector of duck and non_duck
mean_duck_rgb = model.mean(duck_rgb)
mean_non_duck_rgb = model.mean(non_duck_rgb)

# Get the sigma vector of duck and non_duck
sigma_duck_rgb = model.sigma(duck_rgb, mean_duck_rgb)
sigma_non_duck_rgb = model.sigma(non_duck_rgb, mean_non_duck_rgb)

# Predict Likelihood
for i in range (0, org_img.shape[0]):
    for j in range (0, org_img.shape[1]):
        x_array = np.array([org_img[i][j]])
        x_array = x_array.T
        print('Predicting : {:d}/{:d}, {:d}/{:d}'.format(i, org_img.shape[0], j, org_img.shape[1]))
        possibility_of_duck = model.likelihood(x_array, mean_duck_rgb, sigma_duck_rgb)
        possibility_of_non_duck = model.likelihood(x_array, mean_non_duck_rgb, sigma_non_duck_rgb)
        if possibility_of_duck[0] >= possibility_of_non_duck[0]:
            org_img[i][j] = [255, 255, 255]
        else :
            org_img[i][j] = [0, 0, 0]

plt.subplot(2, 1, 1)
plt.title("voc_img")
plt.imshow(voc_img)

plt.subplot(2, 1, 2)
plt.title("predict_img")
plt.imshow(org_img)
plt.show()