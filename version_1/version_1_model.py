import numpy as np

# from voc data find pixels of duck & non_duck
def find_duck_and_non_duck_pixels(voc_img):
    # declare two lists to put in the pixels of labels
    duck_pixels = []
    non_duck_pixels = []
    # find duck and non_duck pixels, then append into the related list
    for i in range(0, voc_img.shape[0]):
        for j in range(0, voc_img.shape[1]):
            print("Collecting and Scanning Pixels : {:d}/{:d}, {:d}/{:d}".format(i, voc_img.shape[0], j, voc_img.shape[1]))
            if (voc_img[i][j] == [0, 0, 128]).all():  # is duck pixels
                a = [i, j]
                duck_pixels.append(a)
            if (voc_img[i][j] == [0, 128, 0]).all():  # is non_duck pixels
                b = [i, j]
                non_duck_pixels.append(b)
    return duck_pixels, non_duck_pixels

# from pixels find the rgb values of duck & non_duck
def find_duck_and_non_duck_rgb(org_img, duck_pixels, non_duck_pixels):
    # declare two lists to put in the rgb values of labels
    duck_rgb = []
    non_duck_rgb = []
    # find duck and non_duck rgb values, then append into the related list
    for i in range(0, len(duck_pixels)):
        print('Get duck RGB : {:d}/{:d}'.format(i, len(duck_pixels)))
        x = duck_pixels[i][0]
        y = duck_pixels[i][1]
        duck_rgb.append(org_img[x][y])
    for i in range(0, len(non_duck_pixels)):
        print('Get non duck RGB : {:d}/{:d}'.format(i, len(non_duck_pixels)))
        x = non_duck_pixels[i][0]
        y = non_duck_pixels[i][1]
        non_duck_rgb.append(org_img[x][y])

    return duck_rgb, non_duck_rgb

# Mean of feature vector (RGB value)
def mean(array):
    mean_feature_vector = [0, 0, 0]
    for i in range (0, len(array)):
        mean_feature_vector += array[i]
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
    sigma_vector = sigma_vector/(len(rgb_array) - 1)
    return sigma_vector

# Predict Likelihood
def likelihood (feature_vector, mean_vector, sigma_vector):
    constant = 1 / ((2*np.pi)**(3/2)*(np.linalg.det(sigma_vector))**(1/2))
    np_linalg_inv = np.linalg.inv(sigma_vector)
    feature_vector_minus_mean_vector = feature_vector - mean_vector
    exponential = np.exp((-0.5) * np.dot(np.dot((feature_vector_minus_mean_vector.T), np_linalg_inv), feature_vector_minus_mean_vector))
    return constant*exponential