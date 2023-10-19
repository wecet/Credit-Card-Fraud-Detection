# Credit Card Fraud Detection with SVM

This code is designed to create a machine learning model for detecting credit card fraud using a Support Vector Machine (SVM) classifier. Below is a report explaining the choices made in the code:

## Data Preprocessing:
1. **Importing Libraries**: The code starts by importing necessary libraries such as Pandas, scikit-learn, imbalanced-learn (for dealing with class imbalance), and Matplotlib for data manipulation, model training, and result visualization.

2. **Data Loading**: The credit card transaction data is loaded from a CSV file ('creditcard.csv') using Pandas. It's important to note that 'creditcard.csv' should contain relevant features, including 'Class' to indicate whether a transaction is fraudulent or not.

3. **Data Cleaning**: The code drops any rows with missing values, ensuring that the data is clean and ready for further processing.

4. **Feature Scaling**: StandardScaler from scikit-learn is used to standardize the features (V1 to V28) to have a mean of 0 and a standard deviation of 1. Standardization is a common practice to make sure that all features have the same scale and to help the SVM algorithm perform better.

## Addressing Class Imbalance:
5. **Random Under-sampling and SMOTE**: Credit card fraud datasets often suffer from class imbalance, where non-fraudulent transactions significantly outnumber fraudulent ones. To mitigate this issue, the code employs a combination of Random Under-sampling and Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset. Random Under-sampling reduces the majority class (non-fraudulent transactions), and SMOTE generates synthetic examples of the minority class (fraudulent transactions).

## Train-Test Split:
6. **Train-Test Split**: The data is split into training and testing sets. 80% of the resampled data is used for training, and 20% is reserved for testing the SVM model. This is a common practice to evaluate a model's performance on unseen data.

## Support Vector Machine (SVM) Classifier:
7. **SVM Classifier**: A linear Support Vector Machine (SVM) classifier is chosen for its capability to create a clear boundary between classes, especially after preprocessing. SVMs are known for their robustness and effectiveness in binary classification tasks.

8. **Model Training**: The SVM classifier is trained on the training data, which includes both the original and synthetic examples.

9. **Model Persistence**: The trained SVM classifier is saved to a file named 'model.sav' using the Pickle library, allowing the model to be easily loaded and used in the future.

## Summary:
In summary, this code presents a robust approach to credit card fraud detection using a Support Vector Machine (SVM) classifier. Several critical choices are made to handle data preprocessing, class imbalance, model selection, and performance evaluation. These choices reflect best practices in the field of fraud detection and machine learning.

However, it's important to note that model evaluation should not solely rely on accuracy, as accuracy can be misleading in imbalanced datasets. Further tuning and experimentation may be required to achieve the desired balance between precision and recall, depending on the specific needs and constraints of the fraud detection system.
