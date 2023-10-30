# Predictive Model for Optimizing Online Credit Card Payments
![cardmapr-nl-s8F8yglbpjo-unsplash](https://github.com/maximkiesel1/model_engineering_online_payment_service/assets/119667336/256390cd-0953-46c9-886a-247c8ad60aa2)

# Introduction
This repository contains the implementation of a predictive model for optimizing online credit card payments for a large, globally operating retail company experiencing reliability issues with online payments. High failure rates of online credit card payments have led to significant financial losses for the company and decreased customer satisfaction.

Online credit card payments are typically processed through specialized service providers called Payment Service Providers (PSPs). The selection of the appropriate PSP for a specific transaction is currently based on a fixed rule set and is done manually. However, it is believed that a data-driven predictive model can lead to more precise and efficient decisions.

Project Organization and Use Case Description
This project follows the CRISP-DM model (Cross-Industry Standard Process for Data Mining), a widely-used procedural model for data mining projects. It offers a structured approach to carrying out data mining projects and ensures that all important aspects of the project are taken into account.

The client for this project is the online payment department of a large retail company. The department is facing the problem that the costs and number of failed transactions are high. The primary reason for this is the static logic used to decide which PSP to use for a particular transaction.

The main goal of the project is to reduce transaction failures and transaction costs. This is achieved by developing a predictive model capable of automatically selecting the most suitable PSP for a particular transaction. The objective is to increase the success rate of transactions while simultaneously minimizing transaction costs.

# Features Explanation
The timestamp "tmsp" is split into several features to better capture the temporal dimension of the data. These split creates the new features "Year", "Month", "Day", "Day of the Week" (numeric series 0 to 6), "Quarter", "On a Weekend" (as binary 0 or 1), and "Hour". This transformation allows the models to better recognize seasonal and cyclic patterns in the data.

The feature "amount" was converted from continuous numerical data to ordinal numerical data by dividing the continuous numerical data into four quantiles. This transformation allows for the effects of transfer amounts on transaction success to be examined not as a continuous value, but in the form of categories.

The feature "country" is omitted, as there is no significant correlation with the target variable "success".

The features "card" and "PSP" are transformed using one-hot encoding. One-hot encoding is a method for converting categorical data into a binary format that can be processed by machine learning models.

# Model Training, Optimization, and Deployment
First, a simple baseline machine learning model is created, in this case, a Random Forest classification algorithm, to check the general performance capability of a model with the existing data and features and to switch a preliminary ML model to production as quickly as possible.

The data is split in an 80/20 ratio into training and test data. Cross-validation is used to obtain a robust estimate of model performance. This model achieves a weighted F1 score of 0.7120, which is a solid performance for a baseline model.

To improve the performance of the Random Forest model, hyperparameter optimization is performed using grid search. The optimized parameters are:
- Number of decision trees
  - 'n_estimators': [50, 100, 200]
- Maximum number of features considered when splitting a node
  - 'max_features': ['sqrt', 'log2']
- Maximum depth of the decision trees
  - 'max_depth': [4,5,6,7,8]
- Criterion for the quality of a split
  - 'criterion': ['gini', 'entropy']

The optimization provides the following parameters:
- 'n_estimators': 200
- 'max_features': 'sqrt'
- 'max_depth': 8
- 'criterion': 'gini'

With the optimized parameters, the F1 score improves to 0.7675.

As an alternative to the Random Forest model, a Gradient Boosting model is tested. The optimization of this model yields the following parameters:
- 'n_estimators': 300
- 'learning_rate': 1
- 'max_depth': 5

With these optimized parameters, the Gradient Boosting model achieves an F1 score of 0.7468.

Based on these results, the optimized Random Forest model is selected for the production environment.

# Conclusion
This work contributes to a better understanding of the challenges and opportunities of optimizing online credit card payments. It shows that data-driven modeling and machine learning can be effective tools in addressing these challenges and offers practical solutions that can be applied in similar contexts.
