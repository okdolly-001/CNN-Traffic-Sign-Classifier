# **Traffic Sign Recognition** 


---


##### Check out my interactive Jupyter notebook:
[![Binder](http://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/okdolly/CNN-Traffic-Sign-Classifier/master?filepath=Traffic_Sign_Classifier.ipynb)

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"




### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.


This bar chart shows the frequency of every traffic sign class.
Highly underrepresented classes:

Class 0: Speed limit (20km/h)

Class 19: Dangerous curve to the left

Class 40: Roundabout mandatory

Class 41: End of no passing

Class 42: End of no passing by vehicles over 3.5 metric tons

The amount of data is uneven, which might bias the model to prefer the overrepresented classes.


![Class Representations](https://user-images.githubusercontent.com/18589970/39525047-bc2818e2-4dcf-11e8-8efa-0ab553825a8c.png)
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 


As a first step, I decided to convert the images to grayscale because the authors in the Lenet paper said that using color channels didn’t seem to improve things a lot.
Next,I normalize the image data with Min-Max scaling to a range of [-0.5,0.5] The image after min-max scaling has a smaller standard deviations, which can suppress the effect of outliers.

Here is an example of a traffic sign image before and after grayscaling.

![preprocess](https://user-images.githubusercontent.com/18589970/39526205-f2daa1fe-4dd2-11e8-87ad-492e279505ad.jpg)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grascale image   							| 
| Convolution  3x3     | 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												        
| Convolution  3x3     | 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												
| Max pooling	 2x2     | 2x2 stride,  outputs 14x14x32 				        |
| Dropout 50%					 |				
| Flatten	     				 |
| Fully connected		| 128       									|
| RELU					|												|
| Fully connected		| 43         									|
| Softmax				| 43        									|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
Here are the model parameters:
* nb_epoch = 10
* batch_size = 128
* nb_classes = 43
* optimizer='adadelta'
* loss='categorical_crossentropy'
* metrics='accuracy'

I train the model for 10 epochs (10 iterations over all samples in the x_train and y_train tensors), in mini-batches of 128 samples. I choose adadelta because its automatic adaptive and decaying learning rate allows it to run fast in the beginning and slow down when we are near the valley. It avoids getting stuck at local minimum.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.55%
* validation set accuracy of 98.09% 
* test set accuracy of 92.05%


My neural architecture went through five stages:

1. Compile a basic sequential model
* Use adam optimizer and categorical_crossentropy loss function.
* Train the network for ten epochs and validate with 20% of the training data.
* Training Accuracy: 0.9303 - Validation Accuracy: 0.8795

2. Convolutions
* Re-construct the previous network
* Add a convolutional layer with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
* Add a ReLU activation after the convolutional layer.
* Training Accuracy: 0.9321 - Validation Accuracy: 0.9285

3. Pooling
* Re-construct the network
* Add a 2x2 max pooling layer immediately following your convolutional layer.
* Training Accuracy: 0.8798 - Validation Accuracy: 0.9181

4.Dropout
* Re-construct the network
* Add a dropout layer after the pooling layer. Set the dropout rate to 50%.
* Training Accuracy: 0.9424 - Validation Accuracy: 0.9570

5. Final model
* Re-construct the network
* Use adadelta optimizer and categorical_crossentropy loss function.
* Adding a Convolution 3x3 layer followed by a Relu activation layer, a Flatten layer and a fully connected layer of 128 units.
* Training Accuracy: 0.9855 - Validation Accuracy: 0.9809


     I put convolutional layers at the front and fully connected (Dense) layer near the end. The convolutional layers (with activation and pooling) learn local patterns by extracting features from an image.  Once we have extracted the features, then we want to make some decisions about what those features mean with respect to what we are trying to classify. Dense layer learns high-level patterns space and acts a decision-maker for the classification problem.

    For pooling operation, I tried both average pooling and max pooling, but max pooling worked better. Features are the spatial presence of some pattern or concept over different regions, thus it is more informative to look at the maximal presence of different features than at their average presence.  So the most reasonable subsampling strategy is to first produce dense layers of features and then look at the maximal activation of the features over small windows, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![1](https://user-images.githubusercontent.com/18589970/39526410-81f82e74-4dd3-11e8-948d-3d9bfc9caf11.jpg)
![2](https://user-images.githubusercontent.com/18589970/39526411-820eee48-4dd3-11e8-9c82-63b9aeef9d0f.jpg)
![3](https://user-images.githubusercontent.com/18589970/39526412-8224bc82-4dd3-11e8-96ec-5a465b5fd16a.jpg)
![4](https://user-images.githubusercontent.com/18589970/39526413-823a76bc-4dd3-11e8-9df5-1fbe9a8dacdc.jpg)
![5](https://user-images.githubusercontent.com/18589970/39526414-8250f900-4dd3-11e8-843b-8e15b9925068.jpg)


The fourth image might be difficult to classify because there are three traffic signs in the scene. We want to output the prediction for the middle one, but the network might not pick up the importance of the position in an image of multiple traffic signs. Other images are relatively easy to classify. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Road work   									| 
| Speed limit (50km/hr)   			| Traffic Signals								|
| Road work					| Road work											|
| No entry		| Traffic Signals      							|
| Vehecles over 3.5 metrix tons prohibited	      		| Traffic signals			 				|

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. The reason is that I downsized the all the images to size 32 x 32 because the model takes input of shape 32x32x3. The downsized images are unintelligible even to my eyes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second to last cell the Ipython notebook.


My test images are extremely blurry and the predictions are very off.


For the first image,the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.12253761         			| Road work    									| 
| 0.10780171     				| Dangerous curve to the left										|
| 0.09129338					| Traffic Signals										|
| 0.08925914    			| Yield						 				|
| 0.08430965				    | 	No passing     							|




For the second image,the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.12253761         	| Road work    			          | 
| 0.10780171     		| Dangerous curve to the left			|
| 0.09129338			| Traffic Signals					|
| 0.08925914    		| Yield						 	|
| 0.08430965			| No passing     				     |



Predictions: 

TopKV2(values=array([[0.12253761, 0.10780171, 0.09129338, 0.08925914, 0.08430965],
       [0.12814397, 0.12498648, 0.12057956, 0.08437131, 0.06793941],
       [0.1570053 , 0.11832445, 0.11699283, 0.06727394, 0.06163668],
       [0.1284184 , 0.12246139, 0.10545856, 0.06967688, 0.06417745],
       [0.15442276, 0.13004853, 0.09524234, 0.07418149, 0.06288607]],
      dtype=float32), indices=array([[25, 19, 26, 13,  9],
       [26, 25, 19, 13,  9],
       [25, 26, 19, 13,  5],
       [26, 19, 25, 13,  9],
       [26, 25, 19, 13, 12]], dtype=int32))
