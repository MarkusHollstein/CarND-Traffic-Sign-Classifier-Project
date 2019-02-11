# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

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
[image4]: ./images/child.jpg "Traffic Sign 1"
[image5]: ./images/uv.jpg "Traffic Sign 2"
[image6]: ./images/vorf.jpg "Traffic Sign 3"
[image7]: ./images/stop.jpg "Traffic Sign 4"
[image8]: ./images/30.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the the function len and shape to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed over the classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I tried running with and without conversion to grayscale and the grayscale images led to better results. It is probably a simplification of the image, that makes training easier without loosing much information.


As a last step, I normalized the image data because it is necessary for good results not to have too high values (I forgot the exact reason, perhaps something with too high gradients...)

 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, relu activation,dropout 0.5, outputs 28x28x6 	|
| 					|												|
| Max pooling	      	| 2x2 stride, same padding  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, relu activation,dropout 0.5, outputs 10x10x16	|
| Max pooling           | 2x2 stride, same padding  outputs 5x5x16 	    |
| flatten		        | outputs 400        									|
| fully connected		| relu activation, dropout 0.5 , outputs 240       									|
| fully connected		| relu activation, dropout 0.5 , outputs 120					|
| fully connected		| relu activation, dropout 0.5 , outputs 83						|
| fully connected       | logits outputs 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with softmax cross entropy with logits, a learning rate of 0.001, batch size of 128 and 40 epochs.
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of .943
* test set accuracy of .935

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started with the LeNet architecture used in the lecture
* What were some problems with the initial architecture?
I had some trouble tuning the parameters in a way to make the model even learn anything. Especially when I tried to use dropout I hardly got more than the "random guessing" 5% of accuracy. Besides the accuracy did not reach the 93% when the model learned.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I used  tf.layers instead of tf.nn. stuff. This helped to be able to use dropout (at least that was my impression) to prevent overfitting. Besides I added another fully connected layer in order to add more complexity to my model.
* Which parameters were tuned? How were they adjusted and why?
I tuned batch size, learning rate, number of epochs and dropout. Most of it was trying and seeing what works. when the validation accuracy was very unstable I decreased the learning rate, when the learning was too slow I increased it. I chose a number of epochs by stopping it when the validation accuracy did not improve anymore
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolutional layers are a good tool to recognize shapes in images, so this is an obvious choice for this problem.


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is really unsharp. The fourth is taken from a strange angle so there might be problems as well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| children crossing 	| keep right   									| 
| No passing   			| Stop										|
| Priority road	        | Priority road									|
| Stop	        		| Right-of-way at the next intersection					 				|
| 30   			        | 30      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is a very low value compared to the validation and test accuracy. I find it hard to explain this appart from the above described difficulties I can not find a good reason for that bad result. Especially as the probabilities for the misclassified images are all close to 1 and the correct classifications do not appear in the 5 most probable classifications. It seems to be due to the images I have chosen because when they are replaced by test or validation images the accuracy is at least 80%. I would be glad to receive suggestions how to improve this. I searched the forums (like https://discussions.udacity.com/t/prediction-on-new-images-gives-an-accuracy-of-0-even-though-the-model-yielded-a-validation-accuracy-of-97-7-a-test-accuracy-of-95-8-why-that/435182/4) but the proposed changes did not improve my accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell under the respective headline.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Keep right   									| 
| .02     				| End of speed limit (80km/h) 										|
| .006					| Speed limit (30km/h)										|
| .001	      			| Speed limit (60km/h)				 				|
| .0007				    | End of all speed and passing limits      							|


For the second image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.        			| Stop  									| 
| .00     				| No entry 										|
| .00					| Roundabout mandatory										|
| .0	      			| Priority road					 				|
| .0				    | Stop      							|
...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


