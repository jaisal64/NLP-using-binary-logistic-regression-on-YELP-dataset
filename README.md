# NLP-using-binary-logistic-regression-on-YELP-dataset

### 1. **Importing Libraries and Setting Constants**
   - Libraries `csv` and `numpy` are imported.
   - Constants `VECTOR_LEN` and `MAX_WORD_LEN` are set to 300 and 64, respectively.

### 2. **Loading and Preprocessing Data**
   - `load_tsv_dataset(file)` function loads raw data from a tsv file and returns reviews and their ratings.
   - `load_feature_dictionary(file)` function creates a map of words to vectors using word2vec embeddings.

### 3. **Trimming Reviews**
   - `trim_reviews(path_to_dataset)` function trims out-of-vocabulary words from the reviews in the dataset.

### 4. **Embedding Reviews**
   - `embed_reviews(trimmed_dataset)` function calculates the average word2vec embedding for each review in the dataset.

### 5. **Saving Data**
   - `save_as_tsv(dataset, filename)` function saves the embedded dataset into a tsv file.

### 6. **Logistic Regression for Sentiment Analysis**
   - Functions like `sigmoid(x)`, `gd(theta, X, y, learning_rate)`, `train(theta, X, y, num_epoch, learning_rate)`, and others are defined for logistic regression.
   - `logistic_reg(formatted_train, formatted_test, metrics_out, num_epochs, learning_rate)` function performs logistic regression, makes predictions, and writes metrics to a file.

### 7. **Data Visualization and Linear Regression (Cookies Dataset)**
   - Various plots such as scatter plots, histograms, and KDE plots are created to visualize the data.
   - Linear regression models are fitted, and residuals are analyzed.
   - Durbin-Watson test is conducted to check for autocorrelation in the residuals.
   - Log transformation is applied to make the response variable normally distributed, and linear regression models are refitted.

### 8. **Statistical Analysis**
   - Multiple linear regression models are fitted using different sets of predictors, and the summary of the models is printed.
   - Plots are created to visualize the relationship between different variables.

### Code Structure
- The code is structured in a way that it first handles data loading and preprocessing, followed by sentiment analysis using logistic regression, and finally, it performs data visualization and statistical analysis on a different dataset (possibly related to cookies based on variable names).


### Conclusion
This code is a combination of two different analyses: sentiment analysis using logistic regression and a statistical and visual analysis of a dataset related to cookies. Ensure that the necessary data files and libraries are available to run this code successfully.
