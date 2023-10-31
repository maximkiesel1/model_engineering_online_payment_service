# Predictive Model for Optimizing Online Credit Card Payments
![cardmapr-nl-s8F8yglbpjo-unsplash](https://github.com/maximkiesel1/model_engineering_online_payment_service/assets/119667336/256390cd-0953-46c9-886a-247c8ad60aa2)

# Introduction Usecase
This repository contains the implementation of a predictive model for optimizing online credit card payments for a fictitious operating retail company experiencing reliability issues with online payments. High failure rates of online credit card payments could led to significant financial losses for the company and decreased customer satisfaction.

Online credit card payments are typically processed through specialized service providers called Payment Service Providers (PSPs). The selection of the appropriate PSP for a specific transaction is currently based on a fixed rule set and is done manually. However, it is believed that a data-driven predictive model can lead to more precise and efficient decisions.

The main goal of the project is to reduce transaction failures and transaction costs. This is achieved by developing a predictive model capable of automatically selecting the most suitable PSP for a particular transaction. The objective is to increase the success rate of transactions while simultaneously minimizing transaction costs.

# EDA
The dataset consists of 50,410 rows and 8 columns with the following features:
- `Unnamed: 0`	int64: Consecutive row number
- `tmsp`	datetime64[ns]: Timestamp of the transfer/transaction
- `country`	object: Country of transfer
- `amount`	int64: Transfer amount
- `success`	int64: if “1”, then the transfer is successful (target variable)
- `PSP`	object: Name of the payment service provider (PSP = payments service provider)
- `3D_secured`	int64: if “1”, then the customer is 3D identified (thus an even more secure online credit card payment)
- `card`	object: Credit card providers (Master, Visa, Diners)

The dataset does not contain any missing values and duplicates are present only in single occurrences.

- `amount`: The highest and lowest transfer amounts are plausible with an average of €202.40 and a standard deviation of €96.27. The data distribution follows a normal distribution. There is no correlation between amount and 3D_secured, indicating no correlation that could distort the ML model.
<img width="1003" alt="Bildschirmfoto 2023-09-20 um 12 19 27" src="https://github.com/maximkiesel1/model_engineering_online_payment_service/assets/119667336/b0fdae9b-478a-4832-8783-07280cbcefcc">

- `success`: Contains only values 0 and 1. There are significantly more failed payment attempts than successful ones (more 0 than 1 values), indicating a failure rate of 80%. This necessitates a balancing strategy for the ML algorithm. The average smaller amounts are successfully transferred. There is a statistical significance that success and 3D_secured are dependent. There is no correlation between amount and success, indicating amount is rather irrelevant for the ML model.
<img width="1025" alt="Bildschirmfoto 2023-09-20 um 12 27 58" src="https://github.com/maximkiesel1/model_engineering_online_payment_service/assets/119667336/92b7d85b-f551-428f-9645-7e2188d29f8b">

- `3D_secured`: Contains only values 0 and 1. The use of 3D identification is low (more 0 than 1 values) with a usage rate of only 24%. However, the success rate of a payment attempt using 3D-Secure is at 85%.
<img width="1025" alt="Bildschirmfoto 2023-09-20 um 12 39 41" src="https://github.com/maximkiesel1/model_engineering_online_payment_service/assets/119667336/0863f2ba-3557-4259-8432-eb00ccb11e9d">

# Features Engineering
The timestamp "tmsp" is split into several features to better capture the temporal dimension of the data. These split creates the new features "Year", "Month", "Day", "Day of the Week" (numeric series 0 to 6), "Quarter", "On a Weekend" (as binary 0 or 1), and "Hour". This transformation allows the models to better recognize seasonal and cyclic patterns in the data.

The feature "amount" was converted from continuous numerical data to ordinal numerical data by dividing the continuous numerical data into four quantiles. This transformation allows for the effects of transfer amounts on transaction success to be examined not as a continuous value, but in the form of categories.

The feature "country" is omitted, as there is no significant correlation with the target variable "success".

The features "card" and "PSP" are transformed using one-hot encoding. One-hot encoding is a method for converting categorical data into a binary format that can be processed by machine learning models.

The amount of features increase from 7 to 15 features.

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
