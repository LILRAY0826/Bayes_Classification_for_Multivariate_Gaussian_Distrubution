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
* **p(x|ωi) :** It is the ***Posterior Probability*** introduced above.
* ***x :*** It is the ***feature vector*** of ***xi***, where i = i...n, composed by a number of features x1,. . .,xk, i.e, **x** = [x1,x2,. . .,xd]^T^ ∈ ***Ｒ^d^***.
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
        conda create -n=labelme python=3.7`
        conda install pyqt
        conda activate labelme
        pip install labelme
        ```
        
    **Start Label :**
    * Open Labelme API
        ```
        labelme
        ```
        
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
        <label.txt path> Path of label.txt that includes all of label             attributes.***
    * Final :
        ![](https://i.imgur.com/vv8yPeu.png)
            
        You will get a voc dataset, including the above file, and the file of **SegmentationClassPNG** include the PNG image that we will utilize later.
            
            
*IV . Model Construction : model.py*
---
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
* ####  Define a funtion to get rgb value (feature vector) of duck & non duck :
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
*V . Run Program : main.py*
---
```python
import cv2
import numpy as np
import model
import matplotlib.pyplot as plt

# Load voc and original image
voc_img = cv2.imread("label_to_voc_dataset/voc_dataset/SegmentationClassPNG/full_duck.png")
org_img = cv2.imread("label_to_voc_dataset/label_dataset/full_duck.jpeg")

'''''''''
# Image cutting for testing, if the dataset is so large that need to spend lots of time
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

# Show the predict result and compare with the original image
plt.subplot(2, 1, 1)
plt.title("voc_img")
plt.imshow(voc_img)

plt.subplot(2, 1, 2)
plt.title("predict_img")
plt.imshow(org_img)
plt.show()
```
