# User Click Behavior Prediction

## Introduction
This project is to participate in the [2019 National University Computer Challenge Big Data Algorithm Competition](http://www.ncccu.org.cn/case1.html). Based on the desensitized user data given by the competition as a training set, the existing CTR prediction project is used to predict the user's click on advertising behavior. Reference for big data algorithm competition.

CTR estimation (Click-Through Rate Prediction) is a key link in Internet computing advertising, and the accuracy of the estimation directly affects the company's advertising revenue. The most used model in CTR estimation is LR (Logistic Regression). LR is a generalized linear model. Compared with the traditional linear model, LR uses Logit transformation to map the function value to the 0 ~ 1 interval. The mapped function value is CTR estimates. LR is a linear model that is easy to parallelize. It is not a problem to process hundreds of millions of training samples. However, the learning ability of linear models is limited. It requires a large number of feature projects to analyze effective features and feature combinations in advance to indirectly enhance the nonlinear learning ability of LR .
Common models include FM (Factorization Machine) / FFM (Field-aware Factorization Machine), which is used to solve the problem of feature combination under sparse data. LR + GBDT (Gradient Boost Decision Tree), which is used to solve the feature combination problem of LR model, Can learn higher-order non-linear feature combinations.

A tensorflow implementation of a series of deep learning methods to predict CTR, including FM, FNN, NFM, Attention-based NFM, Attention-based MLP, Inner-PNN, Out-PNN, CCPM.

## Result
After 10 rounds of training, the auc index can reach 87%, which is much higher than the accuracy rate of integrated learning algorithms such as Bayes or XGBOOST at the beginning.

