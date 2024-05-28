# Assignment 1 : Language Classification with Non-Neural Methods
### Goal
The aim of the assignment was to classify the given sentences in the data all in Roman Script into their respective language
### Idea
- The final submission is the zip file ee1200490.zip, you can access that to get the sense of what I have implemented
- I started with analysing the dataset and just getting a sense of it by observing the distribution
of the languages
- Since we were supposed to use non neural methods for this, I tried using Logistic Regression,
SVM and Multinomial Naive Bayes but I didn’t get decent results from Logistic Regression
and SVM (can also be a case that there was some bug in the code as well) but as soon as I
got some good results from Naive Bayes, I started with word grams as features and used
Multinomial Naive Bayes
- Then I also tried character grams and even the combination of both as features which
gave a huge 10-12 % jump in overall accuracy
- I could see a huge amount of difference in the samples for English and for rest of the
languages so I also did something for that which I have mentioned below
- I first used a dummy model with 4 and 6 character grams as features and used that on
the training data and then I traversed again on the dataset and kept only those samples
which were correctly predicted by the model and used this as my new training datset
- This is a weird idea to me which I just wanted to try and it worked, I just wanted the
data to be as clean as it can be but this could also lead to something like exposure bias in
neural models
- After I have this, I scaled up the samples of the languages which occurred very less
and brought them to some decent percentage relative to English
- I know this might also
sound very wrong but the distribution was very skewed and it wasn’t giving enough
weight to words from these languages instead if even a single word from other languages
is present, it used to output that
- After this I tried many combinations for character as well as word grams and the best
that I ended with was a combination of 4,6 character grams along with 2 word grams

### Result
I ended up securing the highest accuracy in the class for this assignment with Micro F1 score being 98.0% and Macro F1 score being 98.8%


Please let me know if you have anything to ask in the code itself :)
