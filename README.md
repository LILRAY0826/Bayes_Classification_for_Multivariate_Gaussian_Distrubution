***Pattern Recognition - Bayes Classification for Multivariate Gaussian Distribution on Duck Recognized***
===
*I . Introduction*
---
![](https://1.bp.blogspot.com/-wk3geNYjVtc/VYFwnRGf41I/AAAAAAAAxO4/kL9zUSSHzJE/s1600/Bayes_rule.png)
> #### If we make class ***c*** as ***ω*** :
* **Posterior Probability :** the conditional probability of a class ***ω*** given an input feature vector, denoted as ***P(ω\x)***

* **Class Prior Probability :** the probability of a class ***ω*** under the situation that no feature vector is given, donted as ***P(ω)***
* **Predictor Prior Probability :** the probability that a feature vector ***x*** is observed, also called as ***Evidence***, donated as ***P(x)***.
* **Likelihood :** the conditional probability that some feature vector ***x*** is observed in samples of a class ***ω***, donated as ***P(ω\x)***

*II . Multivariate Gaussian Distrubution*
---
### Formula :
![](https://i.imgur.com/jBqf2zE.png)
---
* **p(x|ωi) :** It is the ***Posterior Probability*** introduced above.
* ***x :*** It is the ***feature vector*** of ***x~i~***, where i = in, composed by a number of features x1,. . .,xk, i.e, **x** = [x1,x2,. . .,xd]T ∈ ***Ｒd***.
A ***feature vector*** is a random vector due to the randomness of the feature values.
* ***µi :*** where ***µi*** , which is the **mean vector** (or **mean** for ***Univariate Gaussian***) of the samples in class ***ωi***. 

    ![](https://i.imgur.com/KY87UKa.png)

* ***Σ :***  which is the **covariance matrix** (or ***variances*** for ***Univariate Gaussian***) of the samples in class ***ωi***.

    ![](https://i.imgur.com/n2Nzisk.png)
    
*III . Data Preprocessing : Labelme*
---
### Facility of Software and Hardware :
#### **Hardware** :
* iMac (Retina 4K, 21.5-inch, 2017)
* macOS Big Sur 11.6
* 3.4 GHz four-core Intel Core i5 Processor
* 8 GB 2400 MHz DDR4 Cache
* Radeon Pro 560 4 GB Display Card
#### **Software**
* Pycharm
* Anaconda
* Python Libary
    * Numpy
    * OpenCV
    * Matplotlib
* Labelme

### Data Pre-processing :
* #### Use labelme to annotate the image
    Labelme Github : https://github.com/wkentaro/labelme
    
    **Steps for installing : Open the terminal**
    
    * Create a enviroment for labelme
        ```
        conda create -n=labelme python=3.7
        conda activate labelme
        conda install pyqt
        pip install labelme
        ``` 
        
    **Start Label :**
    * Open Labelme API
        `labelme`
        
    ![](https://i.imgur.com/bsBsXrg.jpg)

    **Convert json file to voc file :**
    * Clone github file :
    
        `$ git clone https://github.com/wkentaro/labelme` 
        **or**
        
        you can clone my git, it's more customlize.
       
       `https://github.com/LILRAY0826/Bayes_Classification_for_Multivariate_Gaussian_Distrubution.git`
    * Enter sementic_segmentation file :
        `$ cd/d D:\chingi\labelme\examples\semantic_segmentation`
    * Write down your class label in label.txt
    ![](https://i.imgur.com/Mclo1mj.png)

    * Start converting :
        `python labelme2voc.py <data> <data_output> --labels <label.txt path>`
        ***<data> Path of label data(json and jpeg)
        <dataoutput> Path of output data of conversion
        <label.txt path> Path of label.txt that includes all of label attributes.***
    * Final :
        ![](https://i.imgur.com/vv8yPeu.png)
            
        You will get a voc dataset, including the above file, and the file of **SegmentationClassPNG** include the PNG image that we will utilize later.
            
            
*IV . Model Construction :*
---
* #### Abstract :           
    ####    *I define my class labels in five kinds, duck, water, grass, road, sand, let's go ahead and observed what's going on.*           
* ####  Define a funtion to get pixels of all class labels:
```python
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
    total_labels = len(duck_pixels)+len(water_pixels)+len(grass_pixels)+len(road_pixels)+len(sand_pixels)
    return duck_pixels, water_pixels, grass_pixels, road_pixels, sand_pixels, total_labels
```
* ####  Define a funtion to get rgb value (feature vector) of all class labels :
```python
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
```
* ####  Define a funtion to compute mean vector µ~i~ :
```python
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
```
* ####  Define a funtion to compute sigma Σ :
```python
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
```    
* ####  Define a funtion to compute Likelihood P(ω|x) :
```python
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
```
 * ####  Define a funtion to compute prior possibility : 
```python
def prior (total_labels, individual_labels):
    if total_labels == 0 :
        return 0
    return float(individual_labels/total_labels)
```
* ####  Define a funtion to compute accuracy(distance of new prior and old one) : 
```python
def accuracy (new_prior, prior):
    if prior == 0:
        return 0
    return 1 - (abs(new_prior-prior)/prior)
```
*V . Run Programming :*
---
* ####  In this case, I set up the epochs for 20 times :  
```python
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
```
*VI . Conclusion :*
---
![](https://i.imgur.com/0YJYCIk.jpg)
#### In predict_img, it clearly indicates the white pixels are duck and the wrong pixels that is not duck's pixel but is annotated as white pixel are decreased.
#### However, there are also lots of pixels is incorret, this model can be more sophiscated and delicated.
*X . Summary :*
---
### 1. Improvement :
*    ***Time Wasting :***
        *    It is possible way that we remove the constant value in Gaussian Distrubution formula for reducing the time of processing. 
        *    Updating the hardware is also a suitable way, we use 4-core CPU in this project, if 8-core, even 16-core, 32-core CPU, the time wasting would be more concise.
*    ***Accurancy :*** 
        *    Increasing the training data maybe is a good idea, but it must be careful that the more training data, the more possibility of overfitting it will happen.   
        *    In version 1 & 2, we can clearly observe that increasing the kinds of class labels from two to five is an available way to enhance accurancy, in the other word, if we do increase the kinds of class label, it would be more precise.
*    ***Customlize :***   
        *    The codes can be more normalize to utilize in other circumstances, because the codes of this project is bulit up from the situation we have already known, if there is a new situation we haven't met, the programming would be unuseful.
 ### 2. What I've learned :  
*    ***Self-Constructed Model :*** 
            In undergraduate, there is nearly not much of opportunity to construct a model by self, we just use the librarys of machine laerning or deep learning in most of situations, for example scikit-learn, tensorflow, pytorch...etc, so learning  how to make a self-constructed model is a achievement for me.
*    ***Confidence :*** 
            Due to the success of this project, it makes me more confident, I'm no longer afraid of the challenge like this, and the experience will be the fertilizer for my life of engineer.
