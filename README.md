# Brain-tumor-classifier-

In this machine learning project, we build a classifier to detect the brain tumor (if any) from the MRI scan images. By now it is evident that this is a binary classification problem. Examples of such binary classification problems are Spam or Not spam, Credit card fraud (Fraud or Not fraud).

# Brain Tumor Classification Dataset
Please download the dataset for brain tumor classification: Brain Tumor Dataset

The images are split into two folders yes and no each containing images with and without brain tumors respectively. There are a total of 253 images.

# Tools and Libraries used
Brain tumor detection project uses the below libraries and frameworks:

Python – 3.x
TensorFlow – 2.4.1
Keras – 2.4.0
Numpy – 1.19.2
Scikit-learn – 0.24.1
Matplotlib – 3.3.4
OpenCV – 4.5.2
Brain Tumor Project Code
Please download the source code of the brain tumor detection project (which is explained below): Brain Tumor Classification Machine Learning Code

# Steps to Develop Brain Tumor Classifier in Machine Learning
Our approach to building the classifier is discussed in the steps:

Perform Exploratory Data Analysis (EDA) on brain tumor dataset
Build a CNN model
Train and Evaluate our model on the dataset
brain tumor prediction results

# The predictions made by the model will be an array with each value being the probability that it predicts the image belongs to that category. So, we take the maximum of all such probabilities and assign the predicted label to that image input.

A confusion matrix is a matrix representation showing how well the trained model predicts each target class with respect to the counts. It contains 4 values in the following format:

TP FN
FP TN

True positive (TP): Target is positive and the model predicted it as positive
False negative (FN): Target is positive and the model predicted it as negative
False positive (FP): Target is negative and the model predicted it as positive
True negative (TN): Target is negative and the model predicted it as negative
The classification report provides a summary of the metrics precision, recall and F1-score for each class/label in the dataset. It also provides the accuracy and how many dataset samples of each label it categorized.



# Summary
In brain tumor classification using machine learning, we built a binary classifier to detect brain tumors from MRI scan images. We built our classifier using transfer learning and obtained an accuracy of 96.5% and visualized our model’s overall performance.
