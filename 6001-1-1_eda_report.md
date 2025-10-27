Number of attempts: 1 out of 2

Grade: pass 

## Overview

### Criterias Met:
- **Loading the Data from a CSV File**  
  You successfully loaded the dataset from the CSV file using `pandas.read_csv()` which is essential for accessing and initializing data for analysis:   

  ```python
  import pandas as pd
  # Load the dataset
  df = pd.read_csv('./data/resale_transactions_categorised.csv')
  display(df.head())
  display(df.info())
  display(df.isnull().sum())
  display(df.describe().T)
  display(df.duplicated().sum())
  display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
  display(df.drop_duplicates(inplace=True))
  display(df.head())
  display(df.info())
  ```

  Initiating data analysis with loading files is critical in ensuring that further operations are based on the initial structure of the dataset you are supposed to work with.

- **Removing Duplicates**  
  You effectively identified and removed duplicate entries from your dataset. Handling duplicates ensures data quality and improves the reliability of subsequent analyses:

  ```python
  import pandas as pd # Load the dataset
  ```

  ```python
  display(df.duplicated().sum())
  ```

  ```python
  display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
  ```

  ```python
  display(df.drop_duplicates(inplace=True))
  ```

  ```python
  display(df.head())
  ```

  Eliminating duplicates prevents data redundancy and enhances the accuracy of your model's predictions by eliminating potential biases or false patterns that could be drawn from repeated entries.

### Criterias Met (Partially):
- **Checking and Converting Data Types**  
  While you made attempts to convert certain columns to appropriate data types, this process was not exhaustive across all applicable features:

  ```python
  # Load the dataset
  df = pd.read_csv('./data/resale_transactions_categorised.csv')
  display(df.info())
  ```

  ```python
  df['year_month'] = pd.to_datetime(df['month'], format='%Y-%m')
  ```

  ```python
  # Convert 'month' column to datetime
  df['year'] = df['year_month'].dt.year
  df['month'] = df['year_month'].dt.month
  ```

  ```python
  df['storey_range'] = df['storey_range'].apply(convert_storey_range)
  ```

  ```python
  # Absolute negative years values
  df['lease_commence_date'] = df['lease_commence_date'].abs()
  ```

  ```python
  df['remaining_lease_months'] = df['remaining_lease'].apply(extract_lease_info)
  ```

  ```python
  # Replace 'FOUR ROOM' with '4 ROOM' in the 'flat_type' column
  df['flat_type'] = df['flat_type'].replace('FOUR ROOM', '4 ROOM')
  ```

  Proper data type conversion is crucial in preventing errors during analysis and ensuring that operations on data elements are performed correctly.

- **Generating Descriptive Statistics**  
  You created descriptive statistics for some features but didn't cover all applicable numeric and categorical columns comprehensively:

  ```python
  display(df.describe().T)
  ```

  ```python
  print(df['price_category'].value_counts())
  ```

  ```python
  df[['remaining_lease', 'remaining_lease_months']].sample(5)
  ```

  ```python
  df.head()
  ```

  ```python
  df.info()
  ```

  ```python
  df['flat_type'].value_counts()
  ```

  ```python
  df['storey_range'].value_counts()
  ```

  Descriptive statistics provide initial insights into the data's distribution, aiding in understanding the foundational properties necessary for modeling.

- **Identifying and Quantifying Outliers**  
  Your analysis of outliers was limited to only certain subsets of data:

  ```python
  df.describe().T
  ```

  ```python
  terrace_outliers = df[(df['flat_type'] == '3 ROOM') & (df['flatm_name'] == "Terrace")]
  ```

  ```python
  display(terrace_outliers.describe().T)
  ```

  Identifying outliers in data helps mitigate their impact on metrics like mean and standard deviation, thus ensuring robust analytical outputs.

- **Handling Missing Values**  
  You addressed missing values in specific columns but lacked a comprehensive approach to handle all missing data effectively:

  ```python
  df.isnull().sum()
  ```

  ```python
  # Fill missing values in 'town_name' column
  df = fill_missing_names(df=df, id_column='town_id', name_column='town_name')
  # Fill missing values in 'flatm_name' column
  df = fill_missing_names(df=df, id_column='flatm_id', name_column='flatm_name')
  ```

  Ensuring missing data is adequately filled or removed prevents data sparsity issues, enabling more informed modeling efforts.

- **Classifying Numeric and Categorical Features**  
  Features were classified into various types, but a clearer narrative is needed to explain the rationale behind feature transformation choices:

  ```python
  # Load the dataset
  df = pd.read_csv('./data/resale_transactions_categorised.csv')
  display(df.head())
  display(df.info())
  display(df.isnull().sum())
  display(df.describe().T)
  display(df.duplicated().sum())
  display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
  display(df.drop_duplicates(inplace=True))
  display(df.head())
  display(df.info())
  ```

  ```python
  # Encode the target column
  df['price_category'] = df['price_category'].map({'Above Median': 1, 'Below Median': 0})
  # Check the distribution of the target column
  display(df['price_category'].value_counts())
  ```

  ```python
  # Define numerical features to be standardized
  numerical_features = ['floor_area_sqm', 'remaining_lease_months', 'lease_commence_date', 'year']
  # Define nominal features to be one-hot encoded
  nominal_features = ['month', 'town_name', 'flatm_name', 'storey_range']
  # Define ordinal features to be ordinally encoded.
  ordinal_features = ['flat_type']
  ```

  Clearly classifying features as numeric or categorical and adapting transformations appropriately helps in facilitating effective modeling strategies.

- **Generating Visualizations**  
  You incorporated visualizations for categorical features but skipped numeric distribution explorations through histograms or boxplots:

  ```python
  # Count the Number of Listings by Month
  month_counts = df['month'].value_counts().sort_index()
  ```

  ```python
  # Create the Line Chart
  plt.figure(figsize=(14, 7))
  line_plot = sns.lineplot(x=month_counts.index, y=month_counts.values, marker='o')
  ```

  ```python
  # Bar chart for flat type with annotations
  plt.figure(figsize=(10, 6))
  count_plot = sns.countplot(x='flat_type', data=df)
  ```

  ```python
  # Bar chart for storey_range with annotations
  plt.figure(figsize=(10, 6))
  count_plot = sns.countplot(x='storey_range', data=df)
  ```

  ```python
  # Bar chart for price category with annotations
  plt.figure(figsize=(10, 6))
  count_plot = sns.countplot(x='price_category', data=df, palette='pastel')
  ```

  Visualizations are crucial for understanding data distributions and relationships between variables, aiding feature exploration and preparation.

### Criterias not fulfilled:
- **Handling Skewed Distributions**  
  There was no assessment or transformation of features with skewed distributions which might result in poor model performance:

  This task involves identifying skewed data patterns and applying transformations like log, square root, or Box-Cox to normalize feature distributions, thus improving model sensitivity and prediction accuracy.
### Extra Effort Recognition:

- **Splitting the Dataset for Effective Model Evaluation**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
  ```
  You demonstrated foresight by splitting the dataset into training, validation, and test sets. This step is critical to ensuring that your classification model can generalize well to unseen data, and provides a structured method for assessing model performance. This extra effort bolsters the reliability and validity of your model's predictions, directly aligning with the problem statement of accurately classifying HDB resale flats price categories.

- **Training and Evaluating Classification Models**:
  ```python
  logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(max_iter=1000))])
  logreg_pipeline.fit(X_train, y_train)
  ```

  ```python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  logreg_val_accuracy = accuracy_score(y_val, y_val_pred)
  ```

  Your approach to not only train, but also evaluate the different classification models including logistic regression, KNN, and decision trees highlights a comprehensive understanding of model effectiveness. By calculating metrics such as accuracy, precision, recall, and F1 score, you ensure the chosen model meets the objective of accurately classifying flat prices relative to the median. Evaluating models against a validation set enhances model selection and tuning, which is commendable in attaining improved prediction accuracy and robustness in classification tasks.
## Things to Work on

1. **Comprehensive Data Type Verification and Conversion**
   - Ensure that you systematically check and convert the data types of all columns in your dataset. While you have addressed some, not all columns have been reviewed for type suitability. Explicitly mention any object-type columns that could be transformed into numeric types, and document these changes to maintain clarity.

2. **Descriptive Statistics for All Features**
   - Expand the coverage of your descriptive statistics to include all numeric and categorical features. While `df.describe().T` provides a good start, ensure to delve into features like 'floor_area_sqm' and 'remaining_lease_months' to uncover any distribution patterns. Providing a comprehensive overview will enrich your analysis and help in feature engineering decisions.

3. **Broader Outlier Analysis**
   - Extend your outlier analysis beyond just the 'Terrace' category for '3 ROOM' flats. Examine all numeric columns for potential outliers using statistical methods (e.g., IQR, Z-score) and provide a rationale for what constitutes an outlier. This will help you understand the data's variance and adjust your model accordingly.

4. **Comprehensive Handling of Missing Values**
   - Consider developing a broader strategy for dealing with missing values beyond the mapping technique applied to `town_name` and `flatm_name`. Evaluate other potential missing data points and consider approaches such as imputation or removal, ensuring that you handle missing data comprehensively across the entire dataset.

5. **Transformation of Skewed Features**
   - Identify skewed distributions in your numeric columns. Utilize transformations like log, square root, or Box-Cox to improve the distribution's symmetry, which can benefit model performance. Since this step is currently missing, reviewing the skewness of your data distributions will be essential.

6. **Clear Classification and Documentation of Features**
   - While your code suggests a classification of features into numerical, nominal, and ordinal, ensure to explicitly document your rationale behind choosing specific pre-processing techniques for each type. Providing a narrative explanation will clarify your decision-making process to others and reinforce your understanding of the feature types.

7. **More Comprehensive Visualizations**
   - Incorporate a broader range of visualizations, especially for numerical features. Utilize histograms or boxplots to explore numerical data distributions, and scatter plots or pair plots to examine relationships between multiple features. Enhancing your visualizations will offer deeper insights into feature interactions and support your classification model development.

By addressing these points, you'll enhance the depth and quality of your exploratory data analysis, ensuring a robust foundation for your classification tasks.
## Extra Recommendations

1. **Data Integrity and Validation**:
   - It's great that you have successfully loaded the dataset and identified duplicates for removal. Moving forward, consider validating whether removing duplicates might affect the integrity of the data, especially if there are genuine repeated entries under different contexts (e.g., a valid resale flat transaction listed more than once due to different transaction conditions or updates).

2. **Handling Missing Values**:
   - While you have encoded missing data in 'town_name' and 'flatm_name' using another column, it's important to ensure accuracy in these replacements. Whenever possible, cross-verify or validate these imputations with another reliable data source to ensure logical consistency in your dataset.

3. **Data Type Conversion**:
   - You did an impressive job converting the 'month' column into a datetime format. Expanding on this practice, ensure other date-related fields are consistently formatted this way if applicable. This enhances analytical robustness by standardizing how dates are handled throughout the analysis.

4. **Feature Engineering**:
   - Your transformation of the 'storey_range' and 'remaining_lease' to numerical values is a good step. Consider expanding feature engineering further by creating additional derived variables that may have predictive relevance, like the age of the development or proximity to key amenities, which might improve insights into pricing.

5. **Examine Potential Outliers**:
   - You've effectively explored outliers in 3-room terrace flats; consider establishing a routine check for outliers across other critical numerical fields like 'floor_area_sqm' or 'lease_commence_date'. This can be instrumental in preemptively identifying data points that could disproportionately influence your model if they result from data entry errors or other anomalies.

Through these enhancements, you can ensure a more comprehensive data analysis process, strengthening the foundation for subsequent model development.
## Code Evaluation

Your submission demonstrates a commendable understanding of data processing and manipulation using Python and Pandas. The approach to handling duplicates, converting date formats, and transforming the data showcases effective data management techniques. You have made excellent use of comments and documentation, which greatly enhances the readability and maintainability of your code. 

Both objectives in your evaluation received high marks for efficiency, readability, adherence to style and conventions, and scalability. This reflects your strong grasp of coding principles and practices.

___

### Commendable Code Practices

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('./data/resale_transactions_categorised.csv')

display(df.head())
display(df.info())
display(df.isnull().sum())
display(df.describe().T)
display(df.duplicated().sum())
display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
display(df.drop_duplicates(inplace=True))
display(df.head())
display(df.info())
```
This snippet demonstrates efficient data loading and preprocessing, using Pandas to handle duplicates effectively. It's well-commented, which aids clarity.

```python
# Function to convert storey_range to ordinal scale by taking the average of the range
def convert_storey_range(storey_range: str) -> float:
    """
    Converts a storey range string into its average numerical value.

    Example:
        convert_storey_range('07 TO 09') -> 8.0
    """
    range_values = storey_range.split(' TO ')
    return (int(range_values[0]) + int(range_values[1])) / 2

# Use the .apply() method to convert the 'storey_range' column with the function
df['storey_range'] = df['storey_range'].apply(convert_storey_range)
```
The function is a smart approach to convert the storey range and is well-documented, which improves understanding.

```python
# Split the data into training (80%) and test-validation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split the test-validation set (20%) into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```
Efficient data splitting using stratification ensures balanced representation in the training and test sets, showcasing a sophisticated understanding of data handling.
___

### Recommendations for Code Improvement

- **Enhance Robustness with Error Handling:** Integrating error-handling mechanisms, such as try-except blocks for functions dealing with file operations or data transformations, could further improve the robustness of your code against unexpected errors.

- **Explore Advanced Data Processing Techniques:** Consider using more advanced techniques like feature scaling or normalization for preprocessing to potentially improve model performance in future projects.

- **Expand Documentation:** While your code is already greatly documented, adding explanatory docstrings for every function or method will increase the accessibility of complex operations for future readers or collaborators.

- **Experiment with Different Machine Learning Models:** Diversifying model choices, such as experimenting with ensemble methods or hyperparameter tuning, can offer insights and potentially improve prediction outcomes as you advance your data science skills.

This constructive feedback aims to help you build on your existing strengths and explore more complex coding challenges confidently. Keep advancing your skills with enthusiasm! 

___