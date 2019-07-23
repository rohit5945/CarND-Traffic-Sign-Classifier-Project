**Traffic Sign Recognition** 
By:Rohit Kukreja

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points

### Submission Files

This project includes

- The notebook `Traffic_Sign_Classifier.ipynb` (and `signames.csv` for completeness)
- `report.html`, the exported HTML version of the python notebook
- A directory `mydata` containing images found on the web
- `README.md`, which you're reading

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


- Number of training examples = 34799
- Number of valid examples= 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

See the result in cell 8, [notebook](https://github.com/lijunsong/udacity-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

---

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code processing images is in cell 9.

Although colors in the traffic sign are important in real world for
people to recoganize different signs, traffic signs are also different
in their shapes and contents. We can ignore colors in this problem
because signs in our training set are differentiable from their
contents and shapes, and the network seems having no problem to learn
just from shapes.

Therefore, My preprocessing phase normalizes images from [0, 255] to
[0, 1], and grayscales it. You can see the grayscale effects in cell
10.
The reason to normalize the image to `[0-1]` rather than `[-1 1]` using pixel-128/128 is because i observed a reduction in accuracy.
See parameter Test Ananysis table below


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The train, valid and test data are prepreocessed in cell 9. I use
`sklearn.train_test_split`  to split training data. The code to split the data
is in function `train` (see cell 15).

To cross validate my model, I randomly split the given training sets
into training set and validation set. I preserved 20% data for
validation.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code is in function `classifier` (see cell 11).

I adapted LeNet architecture: Two convolutional layers followed by one
flatten layer, drop out layer, and three fully connected linear
layers.

1. convolution 1: 32x32x1  -> 28x28x12 -> tanh -> 14x14x12 (pooling)
2. convolution 2: 14x14x12 -> 10x10x25 -> tanh -> 5x5x25   (pooling)
3.       flatten: 5x5x25   -> 625
4.      drop out: 625      -> 625
5.        linear: 625      -> 300 ->relu
6.        linear: 300      -> 150 -relu
7.        linear: 150      -> 43 ->relu

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is in cell 15 and 16.

I train the model in 50 iterations (epochs), and each iteration is
trained with 256 batch size. Adam optimizer is used with learning rate
0.003.
Though I tried different permutations among learning rate,batch size, no of epochs, activation function etc. and analysed the accuracy among each of them

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is in cell 16, 17, 18.

My final model results were:
* training set accuracy of 0.99 (overfitting the cross validation)
* validation set accuracy of 0.937
* test set accuracy of 0.93


The first model is adapted from LeNet architecture. Since LeNet
architecture has a great performance on recognizing handwritings, I
think it would also work on classifying traffic signs.

I used the same parameter given in LeNet lab. Its training accuracy
initially was around 90%.
I captured the accuracy by changing different parameters which are depicted in below table

| Test# | learning rate | #Epochs | batch size | activation function | image normalization | train acc. | validation acc . | test acc. 
| --- | --- | -- | -- | -- | -- | -- | -- | -- | 
| `1` | 0.005 | 10 | 128| relu on all layers| pixel/255 |**99.2** | **93.9**| Not evaluated
| `2` | 0.005 | 50 | 256| relu on all layers| pixel/255 |**99.2** | **95.9**| Not evaluated
| `3` | 0.003 | 50 | 256| relu on all layers| pixel/255 |**99.5** | **96.6**| Not evaluated
| `4` | 0.003 | 50 | 256| relu on **convolution** and **tanh** on full layers| pixel/255 |**99.7** | **95.2**| Not evaluated
| `5` | 0.003 | 50 | 256| relu on **full** and tanh on **convolution** layers| pixel/255 |**99.8** | **96.5**| **93.8**
| `6` | 0.003 | 50 | 256| relu on **full** and tanh on **convolution** layers| pixel-128/128 |**99.7** | **96.0**| **81.6**

Based on above `epoch`, `activation function`,`normality`,`batch_size`, and `rate` parameters, and settled at

- `epoch` 50
- `batch_size` 256
- `learning rate` 0.003
- `activation function`  relu on **full** and tanh on **convolution** layers

The final accuracy in validation set is around **96.4%**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The chosen signs are visualized in cell 20.

I want to see how the classifier performs on similar signs. The
General Caution and Traffic signals: they both look like a vertical bar
(see the visualization) when grayscaled. And pedestrains and child
crossing look similar in low resolution.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is in cell 22. The
result is explained and virtualized in detail in cell 28.

The accuracy on the new traffic signs is **90.9%**, while it was **94.26%** on
the test set. This is a sign of underfitting. By looking at the
virtualized result, I think this can be addressed by using more image
preprocessing techniques on the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the submitted version, the model can correctly guess 9 out of 11
signs. The accuracy is 90.9%. 

By looking at the virtualized data. The predictions of pedestrains,
children crossing, and speed limit 60km/h are actually close
enough. This is actually consistent to my various
experiments. Sometimes the prediction accuracy can be as good as
90%. I think to get the consistent correctness, I need more good
data. One simple thing to do might be to preprocess the image by
brightening dark ones.
