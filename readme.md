# Fraud Detection Case Study

## Premise

You are a contract data scientist/consultant hired by a new e-commerce site to try to weed out fraudsters. The company
unfortunately does not have much data science expertise... so you must properly scope and present your solution to the
manager before you embark on your analysis. Also, you will need to build a sustainable software project that you can
hand off to the companies engineers by deploying your model in the cloud. Since others will potentially use/extend your
code you NEED to properly encapsulate your code and leave plenty of comments.

## Scope

Given highly sensitive data in a json format, it was our task to determine whether the events hosted on our customers
website should be investigated for fraud. We were then tasked to create a web interface in which event data could be
uploaded, run through our model, and checked for possible fraud.

## Model

For this assignment we modeled the data using a random forest classifier. Using this model we were able to achieve a
model with the following metrics:

Accuracy = .988

Precision = .954

Recall = .912

## Try it for Yourself!

Step 1: Clone this repository to your local computer

Step 2: Run form_app.py in debug mode (using pycharm)

Step 3: Copy and paste 'http://0.0.0.0:5000' to your browser of choice


