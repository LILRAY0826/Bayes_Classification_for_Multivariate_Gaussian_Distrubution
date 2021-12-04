import cv2
import numpy as np
import version_1_model
import matplotlib.pyplot as plt

# Load voc and original image
voc_img = cv2.imread("label_to_voc_dataset/voc_dataset/SegmentationClassPNG/full_duck.png")
org_img = cv2.imread("label_to_voc_dataset/label_dataset/full_duck.jpeg")
predict_img = cv2.imread("label_to_voc_dataset/label_dataset/full_duck.jpeg")

'''''''''
# Image cutting for testing, if the dataset is so large that need to spend lots of time
voc_img = voc_img[2000:2500, 2000:2500]
org_img = org_img[2000:2500, 2000:2500]
predict_img = predict_img[2000:2500, 2000:2500]
'''''''''

duck_pixels, non_duck_pixels = version_1_model.find_duck_and_non_duck_pixels(voc_img)
duck_rgb, non_duck_rgb = version_1_model.find_duck_and_non_duck_rgb(org_img, duck_pixels, non_duck_pixels)

print("Training, please wait a second.")
# Get the feature vector of duck and non_duck
mean_duck_rgb = version_1_model.mean(duck_rgb)
mean_non_duck_rgb = version_1_model.mean(non_duck_rgb)

# Get the sigma vector of duck and non_duck
sigma_duck_rgb = version_1_model.sigma(duck_rgb, mean_duck_rgb)
sigma_non_duck_rgb = version_1_model.sigma(non_duck_rgb, mean_non_duck_rgb)

# Predict Likelihood
for i in range (0, org_img.shape[0]):
    for j in range (0, org_img.shape[1]):
        x_array = np.array([org_img[i][j]])
        x_array = x_array.T
        print('Predicting : {:d}/{:d}, {:d}/{:d}'.format(i, org_img.shape[0], j, org_img.shape[1]))
        possibility_of_duck = version_1_model.likelihood(x_array, mean_duck_rgb, sigma_duck_rgb)
        possibility_of_non_duck = version_1_model.likelihood(x_array, mean_non_duck_rgb, sigma_non_duck_rgb)
        if possibility_of_duck[0] >= possibility_of_non_duck[0]:
            predict_img[i][j] = [255, 255, 255]
        else :
            predict_img[i][j] = [0, 0, 0]

# Show the predict result and compare with the original image then save them
plt.subplot(1, 3, 1)
plt.title("org_img")
plt.imshow(org_img)
cv2.imwrite("predict_result/org_img.png", org_img)

plt.subplot(1, 3, 2)
plt.title("voc_img")
plt.imshow(voc_img)
cv2.imwrite("predict_result/voc_img.png", voc_img)

plt.subplot(1, 3, 3)
plt.title("predict_img")
plt.imshow(predict_img)
cv2.imwrite("predict_result/predict_img.png", predict_img)
plt.show()