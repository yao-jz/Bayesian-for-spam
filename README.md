# A simple python implementation of Bayesian classification for Spam

In this project, I use a simple Bayesian classifier for spam identification.

## Dataset

An English E-mail Dataset: https://plg.uwaterloo.ca/~gvcormac/treccorpus06/

./data/: email files (37,822 in total)


./label/index: labels, each row contains a label (spam/ham) and a relative path to corresponding email file

## Motivation

Since advertising words often appear in spam emails, the probability distribution of these words is very different from that of ordinary emails, so according to the frequency of different words in the bag-of-words model, we can roughly judge whether an email is spam or not, and we can see through the experimental results that this method is really useful.

## Implementation

Use bag-of-words model to count the frequency of different words in emails as the classification basis of classifier.

I used a 5-fold cross validation for my model, and the results are as followed:

|Metric|Score|
|---|---|
|Accuracy|0.9870|
|F1|0.9855|
|F0.5|0.9786|
|Precision|0.9921, 0.9771|
|Recall|0.9880, 0.9849|

For precision and recall, the first number is for spam, and the second is for normal email.

## Result Analysis

Since the data of spam and ham in the training set are unbalanced, roughly spam has 24912 letters and ham has 12910 letters, so only looking at the accuracy may not accurately measure the performance of the classifier. So I calculated the f1-score of macro and the corresponding precision and recall here, and it can be seen from the f1-score that the performance of the classifier is better.

## Discussion

In the spam classification problem, we should ensure that the positive example has the highest precision (because normal emails should preferably not be identified as spam), so when measuring the performance of this spam classifier, we should focus more on the precision, so I also used the value of F0.5.

## What's more

Extract the mail server domain name of the sender in the emails, such as gmail.com, yahoo.com, haverford.edu, rpi.edu, and then use the domain information to do the classification.

So, I used a mixture of bag-of-words model and features of the email sender to do the classification and get the following results:

|Metric|Score|
|---|---|
|Accuracy|0.9879|
|F1|0.9865|
|F0.5|0.9795|
|Precision|0.9932, 0.9777|
|Recall|0.9884, 0.9869|

From the results, it is better than using the bag-of-words model alone, indicating that using both features for classification is a better choice.
