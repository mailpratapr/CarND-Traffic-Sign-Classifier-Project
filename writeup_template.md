
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import os

# TODO: Fill this in based on where you saved the training and testing data
path = os.getcwd()
print(path)
training_file = "/home/workspace/data/train.p"
validation_file="/home/workspace/data/valid.p"
testing_file = "/home/workspace/data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

    /home/workspace/CarND-Traffic-Sign-Classifier-Project


---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_valid)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 4410
    Image data shape = (34799, 32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
num_of_samples=[]
plt.figure(figsize=(12, 16.5))
for i in range(0, n_classes):
    plt.subplot(11, 4, i+1)
    x_selected = X_train[y_train == i]
    plt.imshow(x_selected[0, :, :, :]) #draw the first image of each class
    plt.title(i)
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()

#Plot number of images per class
plt.figure(figsize=(12, 4))
plt.bar(range(0, n_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

plt.figure()
hist_train = np.histogram(y_train,n_classes)
plt.plot(np.arange(0,n_classes,1),hist_train[0])
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.figure()
hist_valid = np.histogram(y_valid,n_classes)
plt.plot(np.arange(0,n_classes,1),hist_valid[0])
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.figure()
hist_test = np.histogram(y_test,n_classes)
plt.plot(np.arange(0,n_classes,1),hist_test[0])
plt.xlabel("Class number")
plt.ylabel("Number of images")

print("Min number of images per class =", min(num_of_samples))
print("Max number of images per class =", max(num_of_samples))
```


![png](output/output_8_0.png)



![png](output/output_8_1.png)


    Min number of images per class = 180
    Max number of images per class = 2010



![png](output/output_8_3.png)



![png](output/output_8_4.png)



![png](output/output_8_5.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

1. Converting to grayscale - It also helps to reduce training time, which was nice when a GPU wasn't available. But that doesn't seem to be an efficient approach becase the traffic signs have a lot of color information which could be useful in classifying them. So I decided to use the same image with just normalization.

2. Normalizing the data to the range (-1,1) - This was done using the line of code X_train_normalized = (X_train - 128)/128. The resulting dataset mean wasn't exactly zero, but it was reduced from around 82 to roughly -0.35. 

I chose to do this mostly because it was fairly easy to do. How it helps is a bit nebulous to me, the gist of which is that having a wider distribution in the data would make it more difficult to train using a singlar learning rate. Different features could encompass far different ranges and a single learning rate might make some weights diverge.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
X_train, y_train = shuffle(X_train, y_train)

X_train = X_train / 255 

EPOCHS = 100
BATCH_SIZE = 256
```

# CNN Architecture

| Layer | Input | Output  | Description | Filter Size |
|-------|-------|---------|-------------|--------------|
| CNN-1 | 32x32x3 | 30x30x32  | RGB Image, input image with stride 1 and valid padding |
| CNN-2 | 30x30x32 | 28x28x32  | CNN with stride 1 and valid padding |
| Pooling | 28x28x32 | 14x14x32  | 2x2 max pool |
| CNN-3 | 14x14x32 | 12x12x64  | CNN with stride 1 and valid padding |
| CNN-4 | 12x12x64 | 10x10x64  | CNN with stride 1 and valid padding |
| Pooling | 10x10x64 | 5x5x64  | 2x2 max pool |
| CNN-4 | 5x5x64 | 3x3x128  | CNN with stride 1 and valid padding |
| Flatten | 3x3x128 | 1152  | Flatten the layer |
| FC-1 | 1152 | 1024  | Fully connected Layer |
| ReLU | 1024 | 1024  | Activation |
| DO | 1024 | 1024  | DropOut |
| FC-2 | 1024 | 1024  | Fully connected Layer |
| ReLU | 1024 | 1024  | Activation |
| DO | 1024 | 1024  | DropOut |
| FC-3 | 1024 | 43  | Fully connected Layer |

I started with the LeNet example. That model seems to work well with hand written code recognition and it is proven to be one of the best. Current dataset isn't as easy as MNIST Data, it has complex information and the data is noisy and blurry too. With the LeNet model I could not be able to get the accuracy that I was expecting. So I modied the CNNs to be deeper to improve the results.

That doesn't seem to give me more than 60% accuracy in first 50 epochs. So I started modyfing the Fully connected layers to see if the result changes. Looks like that made a huge impact in the model. I got around 90% accuracy. From the references I see that dropout layers could be something that could make the model more reliable. Which apparently improved the accuracy to around 93%.

Then by adjusting the batch size and learning rate, I could be able to achieve 96-97% accuracy.


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 30x30x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1   = tf.nn.relu(conv1)

    # Layer 2: Convolutional. Output = 28x28x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2   = tf.nn.relu(conv2)

    # Pooling. Input = 28x28x32. Output = 14x14x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Layer 3: Convolutional. Iutput = 14x14x32. Output = 12x12x64
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(64))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3   = tf.nn.relu(conv3)

    # Layer 4: Convolutional. Iutput = 12x12x64. Output = 10x10x64
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(64))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
    conv4   = tf.nn.relu(conv4)

    # Pooling. Input = 10x10x64. Output = 5x5x64.
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # Layer 5: Convolutional. Iutput = 5x5x64. Output = 3x3x128
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(128))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
    conv5   = tf.nn.relu(conv5)

    # Flatten. Input = 3x3x128. Output = 1152.
    fc0   = flatten(conv5)

    # Layer 3: Fully Connected. Input = 1152. Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1152, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 1024. Output = 1024.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(1024))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 1024. Output = 43.
    fc3_W       = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = mu, stddev = sigma))
    fc3_b       = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits
```


```python
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.0009
kp = 0.85
keep_prob = tf.placeholder(tf.float32)
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: kp})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.801
    
    EPOCH 2 ...
    Validation Accuracy = 0.890
    
    EPOCH 3 ...
    Validation Accuracy = 0.933
    
    EPOCH 4 ...
    Validation Accuracy = 0.929
    
    EPOCH 5 ...
    Validation Accuracy = 0.911
    
    EPOCH 6 ...
    Validation Accuracy = 0.943
    
    EPOCH 7 ...
    Validation Accuracy = 0.953
    
    EPOCH 8 ...
    Validation Accuracy = 0.960
    
    EPOCH 9 ...
    Validation Accuracy = 0.953
    
    EPOCH 10 ...
    Validation Accuracy = 0.949
    
    EPOCH 11 ...
    Validation Accuracy = 0.956
    
    EPOCH 12 ...
    Validation Accuracy = 0.957
    
    EPOCH 13 ...
    Validation Accuracy = 0.949
    
    EPOCH 14 ...
    Validation Accuracy = 0.962
    
    EPOCH 15 ...
    Validation Accuracy = 0.966
    
    EPOCH 16 ...
    Validation Accuracy = 0.958
    
    EPOCH 17 ...
    Validation Accuracy = 0.957
    
    EPOCH 18 ...
    Validation Accuracy = 0.951
    
    EPOCH 19 ...
    Validation Accuracy = 0.956
    
    EPOCH 20 ...
    Validation Accuracy = 0.955
    
    EPOCH 21 ...
    Validation Accuracy = 0.954
    
    EPOCH 22 ...
    Validation Accuracy = 0.956
    
    EPOCH 23 ...
    Validation Accuracy = 0.964
    
    EPOCH 24 ...
    Validation Accuracy = 0.942
    
    EPOCH 25 ...
    Validation Accuracy = 0.969
    
    EPOCH 26 ...
    Validation Accuracy = 0.963
    
    EPOCH 27 ...
    Validation Accuracy = 0.952
    
    EPOCH 28 ...
    Validation Accuracy = 0.969
    
    EPOCH 29 ...
    Validation Accuracy = 0.965
    
    EPOCH 30 ...
    Validation Accuracy = 0.967
    
    EPOCH 31 ...
    Validation Accuracy = 0.968
    
    EPOCH 32 ...
    Validation Accuracy = 0.955
    
    EPOCH 33 ...
    Validation Accuracy = 0.962
    
    EPOCH 34 ...
    Validation Accuracy = 0.963
    
    EPOCH 35 ...
    Validation Accuracy = 0.975
    
    EPOCH 36 ...
    Validation Accuracy = 0.961
    
    EPOCH 37 ...
    Validation Accuracy = 0.960
    
    EPOCH 38 ...
    Validation Accuracy = 0.962
    
    EPOCH 39 ...
    Validation Accuracy = 0.950
    
    EPOCH 40 ...
    Validation Accuracy = 0.954
    
    EPOCH 41 ...
    Validation Accuracy = 0.938
    
    EPOCH 42 ...
    Validation Accuracy = 0.929
    
    EPOCH 43 ...
    Validation Accuracy = 0.959
    
    EPOCH 44 ...
    Validation Accuracy = 0.966
    
    EPOCH 45 ...
    Validation Accuracy = 0.969
    
    EPOCH 46 ...
    Validation Accuracy = 0.956
    
    EPOCH 47 ...
    Validation Accuracy = 0.956
    
    EPOCH 48 ...
    Validation Accuracy = 0.970
    
    EPOCH 49 ...
    Validation Accuracy = 0.961
    
    EPOCH 50 ...
    Validation Accuracy = 0.967
    
    EPOCH 51 ...
    Validation Accuracy = 0.963
    
    EPOCH 52 ...
    Validation Accuracy = 0.968
    
    EPOCH 53 ...
    Validation Accuracy = 0.972
    
    EPOCH 54 ...
    Validation Accuracy = 0.961
    
    EPOCH 55 ...
    Validation Accuracy = 0.971
    
    EPOCH 56 ...
    Validation Accuracy = 0.966
    
    EPOCH 57 ...
    Validation Accuracy = 0.966
    
    EPOCH 58 ...
    Validation Accuracy = 0.973
    
    EPOCH 59 ...
    Validation Accuracy = 0.971
    
    EPOCH 60 ...
    Validation Accuracy = 0.937
    
    EPOCH 61 ...
    Validation Accuracy = 0.972
    
    EPOCH 62 ...
    Validation Accuracy = 0.961
    
    EPOCH 63 ...
    Validation Accuracy = 0.953
    
    EPOCH 64 ...
    Validation Accuracy = 0.958
    
    EPOCH 65 ...
    Validation Accuracy = 0.962
    
    EPOCH 66 ...
    Validation Accuracy = 0.959
    
    EPOCH 67 ...
    Validation Accuracy = 0.959
    
    EPOCH 68 ...
    Validation Accuracy = 0.956
    
    EPOCH 69 ...
    Validation Accuracy = 0.980
    
    EPOCH 70 ...
    Validation Accuracy = 0.959
    
    EPOCH 71 ...
    Validation Accuracy = 0.951
    
    EPOCH 72 ...
    Validation Accuracy = 0.956
    
    EPOCH 73 ...
    Validation Accuracy = 0.964
    
    EPOCH 74 ...
    Validation Accuracy = 0.965
    
    EPOCH 75 ...
    Validation Accuracy = 0.957
    
    EPOCH 76 ...
    Validation Accuracy = 0.962
    
    EPOCH 77 ...
    Validation Accuracy = 0.965
    
    EPOCH 78 ...
    Validation Accuracy = 0.965
    
    EPOCH 79 ...
    Validation Accuracy = 0.952
    
    EPOCH 80 ...
    Validation Accuracy = 0.952
    
    EPOCH 81 ...
    Validation Accuracy = 0.974
    
    EPOCH 82 ...
    Validation Accuracy = 0.972
    
    EPOCH 83 ...
    Validation Accuracy = 0.971
    
    EPOCH 84 ...
    Validation Accuracy = 0.972
    
    EPOCH 85 ...
    Validation Accuracy = 0.968
    
    EPOCH 86 ...
    Validation Accuracy = 0.972
    
    EPOCH 87 ...
    Validation Accuracy = 0.942
    
    EPOCH 88 ...
    Validation Accuracy = 0.954
    
    EPOCH 89 ...
    Validation Accuracy = 0.959
    
    EPOCH 90 ...
    Validation Accuracy = 0.962
    
    EPOCH 91 ...
    Validation Accuracy = 0.971
    
    EPOCH 92 ...
    Validation Accuracy = 0.970
    
    EPOCH 93 ...
    Validation Accuracy = 0.959
    
    EPOCH 94 ...
    Validation Accuracy = 0.972
    
    EPOCH 95 ...
    Validation Accuracy = 0.955
    
    EPOCH 96 ...
    Validation Accuracy = 0.966
    
    EPOCH 97 ...
    Validation Accuracy = 0.974
    
    EPOCH 98 ...
    Validation Accuracy = 0.969
    
    EPOCH 99 ...
    Validation Accuracy = 0.976
    
    EPOCH 100 ...
    Validation Accuracy = 0.978
    
    Model saved



```python
with tf.Session() as sess:
    saver.restore(sess, './lenet')
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 0.949


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import csv
import os
import matplotlib.pyplot as plt
import random
import scipy.misc as scms
import numpy as np

%matplotlib inline

signnames = {}
n_images_web = 5
with open('signnames.csv', 'rt') as csvfile:
    signreader = csv.reader(csvfile)
    for k in signreader:
        signnames[k[0]] = k[1]
        
for i in range(0,5):
    index = random.randint(0,len(signnames))
    print(index)
    sample_image = X_test[index]
    resized_image = scms.imresize(sample_image,(32,32,3))
    plt.figure(figsize=(1,1))
    plt.imshow(sample_image)
```

    4
    40
    44
    21
    4



![png](output/output_22_1.png)



![png](output/output_22_2.png)



![png](output/output_22_3.png)



![png](output/output_22_4.png)



![png](output/output_22_5.png)



```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc as scms
import numpy as np

n_images_web = 5

X_test_web = []

dir = 'web_sign/'
names = ['bicyclecrossing.jpg',\
         'nopassing.jpg',\
         'straightorright.jpg',\
         'roadwork.jpg',\
         'childrencrossing.jpg']

Y_test_web = np.array([29,9,36,25,28,])

for name,i in zip(names,range(0,5)):
    image = mpimg.imread(dir+name)
    print("Image {} = {}".format(i,name))
    plt.figure()
    plt.imshow(image)
    print(image.shape)
    # Resize to 32x32 and store in the array of test data
    resized_image = scms.imresize(image,(32,32,3))
    X_test_web.append(resized_image)
```

    Image 0 = bicyclecrossing.jpg
    (470, 450, 3)
    Image 1 = nopassing.jpg
    (656, 665, 3)
    Image 2 = straightorright.jpg
    (470, 450, 3)
    Image 3 = roadwork.jpg
    (225, 300, 3)
    Image 4 = childrencrossing.jpg
    (320, 450, 3)



![png](output/output_23_1.png)



![png](output/output_23_2.png)



![png](output/output_23_3.png)



![png](output/output_23_4.png)



![png](output/output_23_5.png)


The images used for testing appears to be more easily distinguishable than quite a few images from the original dataset. And the images tend to be quite a bit brighter and might occupy a different range in the color space, possibly a range that the model was not trained on. In addition, the German Traffic signs dataset contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches. 

# Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
prediction = tf.argmax( logits, 1 )

# Redeclare in case you want to run this cell alone.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, './lenet')
    output = sess.run(prediction, feed_dict={
        x: X_test_web, 
        keep_prob: 1.0})
    
print("Output classes:")
print("Expected: \n\
       29 (bicycles crossing), \n\
       9 (no passing), \n\
       36 (go straight or right), \n\
       25 (road work), \n\
       28 (children crossing)")
print('Actual:')
print(output)
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Output classes:
    Expected: 
           29 (bicycles crossing), 
           9 (no passing), 
           36 (go straight or right), 
           25 (road work), 
           28 (children crossing)
    Actual:
    [24  9 36 25 28]


The test image 1 is a clear image, high contrast. This will make sure that the model is performing well or not on new data which the model had never seen. This will help in understanding the performance of the model. The accuracy of the model should not get affected by the brightness and contrast. But looks like it does on some data like computer generated images.

# Accuracy


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess,'./lenet')
    accuracy = sess.run(accuracy_operation, feed_dict={
            x: X_test_web, 
            y: Y_test_web,
            keep_prob: 1.0})
    print( "Accuracy = {}%".format( accuracy*100. ) )
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Accuracy = 80.0000011920929%


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
top_k_getter = tf.nn.top_k(tf.nn.softmax(logits),5)

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, './lenet')
    output = sess.run(top_k_getter, feed_dict={
        x: X_test_web, 
        keep_prob: 1.0})

print("Top softmax probabilities:")

for i in range(5):
    print()
    print( "{}:".format( names[i] ) )
    print( "Probabilities")
    print(output.values[i])
    print( "Corresponding labels")
    print(output.indices[i])
    plt.figure()
    plt.bar(output.indices[i],output.values[i], align='center', alpha=0.5)
    plt.xlim([0,42])
    plt.title("{}, label = {}".format( names[i], Y_test_web[i] ) )
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Top softmax probabilities:
    
    bicyclecrossing.jpg:
    Probabilities
    [  1.00000000e+00   6.80405895e-21   0.00000000e+00   0.00000000e+00
       0.00000000e+00]
    Corresponding labels
    [24 29  0  1  2]
    
    nopassing.jpg:
    Probabilities
    [ 1.  0.  0.  0.  0.]
    Corresponding labels
    [9 0 1 2 3]
    
    straightorright.jpg:
    Probabilities
    [ 1.  0.  0.  0.  0.]
    Corresponding labels
    [36  0  1  2  3]
    
    roadwork.jpg:
    Probabilities
    [ 1.  0.  0.  0.  0.]
    Corresponding labels
    [25  0  1  2  3]
    
    childrencrossing.jpg:
    Probabilities
    [ 1.  0.  0.  0.  0.]
    Corresponding labels
    [28  0  1  2  3]



![png](output/output_32_1.png)



![png](output/output_32_2.png)



![png](output/output_32_3.png)



![png](output/output_32_4.png)



![png](output/output_32_5.png)


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```

# Results

1. one more add on would be training on a high dimensional dataset (64x64x3) will give more clarity than 32x32x3 dataset
2. Testing out this on a video feed on realtime to get the feel of the application
