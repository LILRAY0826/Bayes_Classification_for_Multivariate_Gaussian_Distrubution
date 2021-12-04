import numpy as np

# from voc data find pixels of all class labels
def find_pixels(voc_img):
    # declare two lists to put in the pixels of labels
    duck_pixels = []
    water_pixels = []
    grass_pixels = []
    road_pixels = []
    sand_pixels = []
    # find duck and non_duck pixels, then append into the related list
    for i in range(0, voc_img.shape[0]):
        for j in range(0, voc_img.shape[1]):
            print("Collecting and Scanning Pixels : {:d}/{:d}, {:d}/{:d}".format(i, voc_img.shape[0], j, voc_img.shape[1]))
            if (voc_img[i][j] == [0, 0, 128]).all():  # is duck pixels BGR
                a = [i, j]
                duck_pixels.append(a)
            elif (voc_img[i][j] == [0, 128, 128]).all():  # is water pixels
                b = [i, j]
                water_pixels.append(b)
            elif (voc_img[i][j] == [128, 0, 0]).all():  # is grass pixels
                c = [i, j]
                grass_pixels.append(c)
            elif (voc_img[i][j] == [128, 0, 128]).all():  # is road pixels
                d = [i, j]
                road_pixels.append(d)
            elif (voc_img[i][j] == [128, 128, 0]).all():  # is sand pixels
                f = [i, j]
                sand_pixels.append(f)
    return duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels

# from pixels find the rgb values of all class labels
def find_rgb(org_img, duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels):
    # declare two lists to put in the rgb values of labels
    duck_rgb = []
    water_rgb = []
    grass_rgb = []
    road_rgb = []
    sand_rgb = []
    # find duck and non_duck rgb values, then append into the related list
    for i in range(0, len(duck_pixels)):
        print('Get duck RGB : {:d}/{:d}'.format(i, len(duck_pixels)))
        x = duck_pixels[i][0]
        y = duck_pixels[i][1]
        duck_rgb.append(org_img[x][y])
    for i in range(0, len(water_pixels)):
        print('Get non duck RGB : {:d}/{:d}'.format(i, len(water_pixels)))
        x = water_pixels[i][0]
        y = water_pixels[i][1]
        water_rgb.append(org_img[x][y])
    for i in range(0, len(grass_pixels)):
        print('Get non duck RGB : {:d}/{:d}'.format(i, len(grass_pixels)))
        x = grass_pixels[i][0]
        y = grass_pixels[i][1]
        grass_rgb.append(org_img[x][y])
    for i in range(0, len(road_pixels)):
        print('Get non duck RGB : {:d}/{:d}'.format(i, len(road_pixels)))
        x = road_pixels[i][0]
        y = road_pixels[i][1]
        road_rgb.append(org_img[x][y])
    for i in range(0, len(sand_pixels)):
        print('Get non duck RGB : {:d}/{:d}'.format(i, len(sand_pixels)))
        x = sand_pixels[i][0]
        y = sand_pixels[i][1]
        sand_rgb.append(org_img[x][y])

    return duck_rgb, water_rgb, grass_rgb, road_rgb, sand_rgb

# Mean of feature vector (RGB value)
def mean(array):
    mean_feature_vector = [0, 0, 0]
    for i in range (0, len(array)):
        mean_feature_vector += array[i]
    if len(array) == 0:
        mean_feature_vector[0] = 0
        mean_feature_vector[1] = 0
        mean_feature_vector[2] = 0
    else :
        mean_feature_vector[0] = mean_feature_vector[0] / len(array)
        mean_feature_vector[1] = mean_feature_vector[1] / len(array)
        mean_feature_vector[2] = mean_feature_vector[2] / len(array)
    mean_feature_vector = np.array([mean_feature_vector])
    return mean_feature_vector.T

# Sigma of feature vector
def sigma(rgb_array, mean_array):
    sigma_vector = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]
    for i in range (0, len(rgb_array)):
        x_array = np.array([rgb_array[i]])
        x_array = x_array.T
        sigma_vector += (x_array - mean_array) * ((x_array - mean_array).T)
    if len(rgb_array) != 0:
        sigma_vector = sigma_vector / (len(rgb_array))
    return sigma_vector

# Predict Likelihood
def likelihood (feature_vector, mean_vector, sigma_vector):
    if np.linalg.det(sigma_vector) != 0 :
        constant = 1 / ((2 * np.pi) ** (3 / 2) * (np.linalg.det(sigma_vector)) ** (1 / 2))
    else :
        constant = 0
    sigma_vector = np.array(sigma_vector)
    if sigma_vector.all() != 0 :
        np_linalg_inv = np.linalg.inv(sigma_vector)
        feature_vector_minus_mean_vector = feature_vector - mean_vector
        exponential = np.exp((-0.5) * np.dot(np.dot((feature_vector_minus_mean_vector.T), np_linalg_inv),feature_vector_minus_mean_vector))
    else :
        exponential = 0
    return constant*exponential