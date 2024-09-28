# Loan Prediction System Using AI

## Overview
This program predicts loan eligibility based on various features extracted from a dataset. It employs machine learning techniques, particularly the Support Vector Machine (SVM) algorithm, to assess whether an applicant qualifies for a loan. The program processes data, visualizes relationships, and makes predictions based on user input.

## Key Features

1. **Data Collection and Processing**:
   - The program begins by loading a dataset using Pandas and performing preliminary data analysis.
   - Key operations include checking for missing values, dropping any rows with missing data, and label encoding categorical variables.

   ```python
   loan_dataset = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
   loan_dataset.dropna(inplace=True)
   loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
   ```

2. **Data Visualization**:
   - The program utilizes Seaborn to create visualizations, providing insights into how various features relate to loan status. This includes counts of loan status based on education, marital status, gender, and more.

   ```python
   sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
   ```

3. **Feature Engineering**:
   - Categorical columns are converted to numerical values to prepare for model training.
   - The dataset is split into features (data) and labels (loan status).

   ```python
   data = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
   label = loan_dataset['Loan_Status']
   ```

4. **Model Training and Evaluation**:
   - The dataset is divided into training and testing sets using `train_test_split`.
   - An SVM classifier is trained on the training data, and its accuracy is evaluated on both the training and testing sets.

   ```python
   classifier = svm.SVC(kernel='linear')
   classifier.fit(X_train, Y_train)
   ```

5. **Making Predictions**:
   - The program collects user input regarding the applicant's details and prepares this data for prediction.
   - Finally, it outputs whether the applicant is eligible for a loan based on the model's prediction.

   ```python
   input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
   prediction = classifier.predict(input_data_reshape)
   ```

## Libraries Used
- **NumPy**: Used for numerical operations and array handling, particularly in reshaping input data for predictions.
- **Pandas**: Essential for data manipulation and analysis, enabling easy loading, cleaning, and processing of the dataset.
- **Seaborn**: Utilized for data visualization, helping to uncover patterns and relationships within the dataset.
- **Scikit-Learn**: Provides tools for model training, evaluation, and various machine learning algorithms, including SVM.

## Conclusion
This loan prediction system effectively combines data processing, visualization, and machine learning to assess loan eligibility. The use of libraries like Pandas and NumPy enhances data handling capabilities, while Seaborn provides valuable visual insights. Future improvements could include refining user input validation and exploring additional machine learning algorithms for better prediction accuracy.
