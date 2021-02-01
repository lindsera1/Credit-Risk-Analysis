# Credit-Risk-Analysis

## Analysis Overview

The purpose of this analysis was to use supervised machine learning models and classification systems to predict credit risk for loan applicants. The big question
here is who is trustworthy, and who is not? We used models such as Logistic Regression, Support Vector Machines, and Random Forests to see which ones performed 
better and with enough precision to use for the prediction of credit risk. Below we haved the detailed performance of each machine learning model, and the 
classification systems we used.


## Results

+ In order to compensate in sample discrepencies that may cause our models have extremely high accuracy, but low precision, we used a variety of undersampling and 
oversampling techniques. This typically happens when the training set has a disproportionately large amount of positives compared to the negatives, so the model 
just overpredicts positive results, which would mean bad loan applications get marked as good. We used Random Oversampling and SMOTE oversampling. Random 
oversampling is where instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. 
SMOTE Oversampling is synthetic in nature, creating new samples based on the value of a randomly chosen points neighbors.

![random](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.26.53%20AM.png)

![smote](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.27.37%20AM.png)

+ The next sampling technique we used was an undersampling technique called ClusterCentroids. It's very similar to SMOTE in that the algorithm identifies clusters
of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down 
to the size of the minority class. The results for this sampling technique are shown.

![cluster](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.28.38%20AM.png)

+ The last sampling technique we used was a combination of SMOTE and ENN, called SMOTEENN. It oversamples the minority class with SMOTE, then cleans the resulting 
data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped. This is how this 
strategy performed.

![smoteenn](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.29.25%20AM.png)

+ The next analysis used classifiers, or ensemble algorithms, called Balanced Random Forest Classifiers, and Easy Ensemble AdaBoost. Random Forest algorithms are 
nice because they bypass the weakness of single tree decision making algorithms, which are usually called weak learners. They're weak because it's one decision tree 
based on a small subset of data. If we combine many of the trees together however, we get a more robust classification system called Random Forest, which are a 
bunch of smaller trees trained on the same data set. In the AdaBoost (adaptive boosting) algorithm, a model is trained then evaluated. After evaluating the errors 
of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting 
is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated 
until the error rate is minimized. Here are there results

![random forest](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.30.29%20AM.png)

![adaboost](https://github.com/lindsera1/Credit-Risk-Analysis/blob/main/Screen%20Shot%202021-02-01%20at%2012.31.04%20AM.png)

## Summary

In summary, as with most machine learning models, classifiers, and resampling techniques, we'll see small, but nonetheless significant variance in the performance 
between them. Compared to all of these, the SMOTEENN resampling technique produced 72 percent accuracy, which did seem to outweigh the results of the other 
approaches.
