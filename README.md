***Pattern Recognition - Bayes Classification for Multivariate Gaussian Distrubution on Duck Recognized***
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
* **p(x|ω~i~) :** It is the ***Posterior Probability*** introduced above.
* ***x :*** It is the ***feature vector*** of ***x~i~***, where i = i~n, composed by a number of features x~1~,. . .,x~k~, i.e, **x** = [x~1~,x~2~,. . .,x~d~]^T^ ∈ ***Ｒ^d^***.
A ***feature vector*** is a random vector due to the randomness of the feature values.
* ***µ~i~ :*** where ***µ~i~*** , which is the **mean vector** (or **mean** for ***Univariate Gaussian***) of the samples in class ***ω~i~***. 

    ![](https://i.imgur.com/KY87UKa.png)

* ***Σ :***  which is the **covariance matrix** (or ***variances*** for ***Univariate Gaussian***) of the samples in class ***ω~i~***.

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
            
            
*IV . Model Construction Version 1:*
---
* #### Abstract :          
    ####    *In version 1, I define my class labels in two kinds, duck & non_duck, so let's go ahead and observed what's going on.*           
* ####  Define a funtion to get pixels of duck & non duck :
```python
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
```
* #### Define a funtion to get rgb value (feature vector) of duck & non duck :
```python
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
```
* ####  Define a funtion to compute mean vector µ~i~ :
```python
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
    sigma_vector = sigma_vector/(len(rgb_array) - 1)
    return sigma_vector  
```    
* ####  Define a funtion to compute Likelihood P(ω|x) :
```python
# Predict Likelihood
def likelihood (feature_vector, mean_vector, sigma_vector):
    constant = 1 / ((2*np.pi)**(3/2)*(np.linalg.det(sigma_vector))**(1/2))
    np_linalg_inv = np.linalg.inv(sigma_vector)
    feature_vector_minus_mean_vector = feature_vector - mean_vector
    exponential = np.exp((-0.5) * np.dot(np.dot((feature_vector_minus_mean_vector.T), np_linalg_inv), feature_vector_minus_mean_vector))
    return constant*exponential
```
*V . Run Programming Version 1 :*
---
```python
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
```

*VI . Conclusion Version 1 :*
---
![](https://i.imgur.com/Zgoj9iI.png)
            
#### In voc_img, the fields of blue are ducks, th fields of greem are non_duck. 
#### You can not only observe easily that all the ducks were roughly detected, but also aware that the road, sand or the others not duck's pixels are detected, so I improved my model for version 2.
            
*VII . Model Construction Version 2 :*
---
* #### Abstract :         
    ####    *In version 2, I define my class labels in five kinds, duck, water, grass, road, sand, so again,  let's go ahead and observed what's going on.*           
* ####  Define a funtion to get pixels of all class labels:
```python
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
*VIII . Run Programming Version 2:*
---
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
cv2.imwrite("predict_result/predict_img.png", predict_img)
plt.show()
```
*IX . Conclusion Version 2:*
---
![](https://i.imgur.com/daI2Nf3.jpg)
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
