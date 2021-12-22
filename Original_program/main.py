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
voc_img = voc_img[2500:3000, 2000:2500]
org_img = org_img[2500:3000, 2000:2500]
predict_img = predict_img[2500:3000, 2000:2500]
'''''''''

# Get all label pixels
duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels, total_pixels = version_2_model.find_pixels(voc_img)
duck_rgb, water_rgb, grass_rgb, road_rgb, sand_rgb = version_2_model.find_rgb(org_img, duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels)

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

likelihood_of_duck = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
likelihood_of_water = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
likelihood_of_grass = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
likelihood_of_road = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
likelihood_of_sand = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)

# Calculate Likelihood
for i in range (0, org_img.shape[0]):
    for j in range (0, org_img.shape[1]):
        x_array = np.array([org_img[i][j]])
        x_array = x_array.T
        print('Calculating Likelihood : {:d}/{:d}, {:d}/{:d}'.format(i+1, org_img.shape[0], j+1, org_img.shape[1]))
        likelihood_of_duck[i][j] = version_2_model.likelihood(x_array, mean_duck_rgb, sigma_duck_rgb)
        likelihood_of_water[i][j] = version_2_model.likelihood(x_array, mean_water_rgb, sigma_water_rgb)
        likelihood_of_grass[i][j] = version_2_model.likelihood(x_array, mean_grass_rgb, sigma_grass_rgb)
        likelihood_of_road[i][j] = version_2_model.likelihood(x_array, mean_road_rgb, sigma_road_rgb)
        likelihood_of_sand[i][j] = version_2_model.likelihood(x_array, mean_sand_rgb, sigma_sand_rgb)

epochs = 20
posterior_of_duck = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
posterior_of_water = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
posterior_of_grass = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
posterior_of_road = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
posterior_of_sand = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
duck_counter = 0
water_counter = 0
grass_counter = 0
road_counter = 0
sand_counter = 0
duck_accuracy = 0

# Training
for epoch in range(0, epochs):
    if epoch == 0:
        # Get the initial prior of duck and non_duck
        prior_of_duck = version_2_model.prior(total_pixels, len(duck_pixels))
        prior_of_water = version_2_model.prior(total_pixels, len(water_pixels))
        prior_of_grass = version_2_model.prior(total_pixels, len(grass_pixels))
        prior_of_road = version_2_model.prior(total_pixels, len(road_pixels))
        prior_of_sand = version_2_model.prior(total_pixels, len(sand_pixels))

    else:
        prior_of_duck_new = float(duck_counter / (org_img.shape[0] * org_img.shape[1]))
        prior_of_water_new = float(water_counter / (org_img.shape[0] * org_img.shape[1]))
        prior_of_grass_new = float(grass_counter / (org_img.shape[0] * org_img.shape[1]))
        prior_of_road_new = float(road_counter / (org_img.shape[0] * org_img.shape[1]))
        prior_of_sand_new = float(sand_counter / (org_img.shape[0] * org_img.shape[1]))
        duck_accuracy = version_2_model.accuracy(prior_of_duck_new, prior_of_duck)
        prior_of_duck = prior_of_duck_new
        prior_of_water = prior_of_water_new
        prior_of_grass = prior_of_grass_new
        prior_of_road = prior_of_road_new
        prior_of_sand = prior_of_sand_new

    for i in range(0, org_img.shape[0]):
        for j in range(0, org_img.shape[1]):
            print("Training epoch : {:d}/{:d} ----> {:d}/{:d}, {:d}/{:d} ----> Accuracy : {:.3f}%".format(epoch+1, epochs, i+1, org_img.shape[0], j+1, org_img.shape[1], duck_accuracy*100))
            posterior_of_duck[i][j] = likelihood_of_duck[i][j] * prior_of_duck
            posterior_of_water[i][j] = likelihood_of_water[i][j] * prior_of_water
            posterior_of_grass[i][j] = likelihood_of_grass[i][j] * prior_of_grass
            posterior_of_road[i][j] = likelihood_of_road[i][j] * prior_of_road
            posterior_of_sand[i][j] = likelihood_of_sand[i][j] * prior_of_sand

            if max(posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]) == posterior_of_duck[i][j]:
                duck_counter += 1
            elif max(posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]) == posterior_of_water[i][j]:
                water_counter += 1
            elif max(posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]) == posterior_of_grass[i][j]:
                grass_counter += 1
            elif max(posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]) == posterior_of_road[i][j]:
                road_counter += 1
            elif max(posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]) == posterior_of_sand[i][j]:
                sand_counter += 1

# Plot the image
for i in range (0, org_img.shape[0]):
    for j in range (0, org_img.shape[1]):
        print('Prepare for showing images : {:d}/{:d}, {:d}/{:d}'.format(i+1, org_img.shape[0], j+1, org_img.shape[1]))
        if max([posterior_of_duck[i][j], posterior_of_water[i][j], posterior_of_grass[i][j], posterior_of_road[i][j], posterior_of_sand[i][j]]) == posterior_of_duck[i][j] :
            predict_img[i][j] = [255, 255, 255]
        else:
            predict_img[i][j] = [0, 0, 0]

# Show the predict result and compare with the original image then save them
plt.subplot(1, 3, 1)
plt.title("org_img")
plt.imshow(org_img)
cv2.imwrite("predict_result/org_img.jpg", org_img)

plt.subplot(1, 3, 2)
plt.title("voc_img")
plt.imshow(voc_img)
cv2.imwrite("predict_result/voc_img.png", voc_img)

plt.subplot(1, 3, 3)
plt.title("predict_img")
plt.imshow(predict_img)
cv2.imwrite("predict_result/predict_img.png", predict_img)
plt.show()
