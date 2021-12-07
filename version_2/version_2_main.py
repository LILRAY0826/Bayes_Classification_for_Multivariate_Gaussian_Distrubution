import cv2
import numpy as np
import version_2_model
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

duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels = version_2_model.find_pixels(voc_img)
duck_rgb, water_rgb, grass_rgb, road_rgb, sand_rgb = version_2_model.find_rgb(org_img, duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels)

print("Training, please wait a second.")
# Get the feature vector of duck and non_duck
mean_duck_rgb = version_2_model.mean(duck_rgb)
mean_water_rgb = version_2_model.mean(water_rgb)
mean_grass_rgb = version_2_model.mean(grass_rgb)
mean_road_rgb = version_2_model.mean(road_rgb)
mean_sand_rgb = version_2_model.mean(sand_rgb)

# Get the sigma vector of duck and non_duck
sigma_duck_rgb = version_2_model.sigma(duck_rgb, mean_duck_rgb)
sigma_water_rgb = version_2_model.sigma(water_rgb, mean_water_rgb)
sigma_grass_rgb = version_2_model.sigma(grass_rgb, mean_grass_rgb)
sigma_road_rgb = version_2_model.sigma(road_rgb, mean_road_rgb)
sigma_sand_rgb = version_2_model.sigma(sand_rgb, mean_sand_rgb)

# Predict Likelihood
for i in range (0, org_img.shape[0]):
    for j in range (0, org_img.shape[1]):
        x_array = np.array([org_img[i][j]])
        x_array = x_array.T
        print('Predicting : {:d}/{:d}, {:d}/{:d}'.format(i, org_img.shape[0], j, org_img.shape[1]))
        possibility_of_duck = version_2_model.likelihood(x_array, mean_duck_rgb, sigma_duck_rgb)
        possibility_of_water = version_2_model.likelihood(x_array, mean_water_rgb, sigma_water_rgb)
        possibility_of_grass = version_2_model.likelihood(x_array, mean_grass_rgb, sigma_grass_rgb)
        possibility_of_road = version_2_model.likelihood(x_array, mean_road_rgb, sigma_road_rgb)
        possibility_of_sand = version_2_model.likelihood(x_array, mean_sand_rgb, sigma_sand_rgb)
        if max([possibility_of_duck, possibility_of_water, possibility_of_grass, possibility_of_road, possibility_of_sand]) == possibility_of_duck :
            predict_img[i][j] = [255, 255, 255]
        else:
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
cv2.imwrite("predict_result/predict_img.jpg", predict_img)
plt.show()
