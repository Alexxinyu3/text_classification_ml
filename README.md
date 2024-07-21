# text_classification_ml
# 1. Introduction
This research aims to utilize machine learning algorithms to construct a text classification model that can distinguish between negative and positive sentiment data. Specifically, we will train a binary text classification model to achieve this goal.

# 2. data preparation
data set: data_for_model

# 3. data_clean

This step is identical to txet_classification_bret--Data_clean. The data cleaning process involves the following steps:

· Removing HTML tags

· Removing special characters and punctuation

· Converting text to lowercase

· Removing stopwords

· Lemmatizing words

· Removing extra spaces

output：Save the cleaned data to a specified path and name it 'cleaned_data_for_model.xlsx'

# 4. feature extraction

Run 'Feature_word2vec.py'

Here using word2vec algorithm converts text data into numerical vectors, which are then used to train the models.

output: Save the results to the data folder and name it 'data_for_model.csv'

# 5. train model

Run 'main.py'

output: In the 'output/text_clf' path you get the following three folders:

· 'metrics' :Contains the following properties of ten machine learning models: 'Method,TPR,FPR,Precision,Recall,ROC AUC,PR AUC,Confusion Matrix'

· 'models' :Contains ten trained machine learning models,they are -SVC,-RandomForest,-LogisticRegression,-KNN,-GaussianNB,-Bagging,-ExtraTree,-DecisionTree,-GradientBoosting,-HistGradientBoosting

· 'plot' :Contains loss and accuracy curves,they are'pr.png'and'roc.png'

# 6. inference
Using 'Prediction_file',after data cleaning and feature extraction, save the data to 'data/data_for_preds.csv'

Run 'inference.py'

output:Save the predictions of the ten models to 'output/predictions.csv'

# 7. data visualization

Run 'data visualization.py'

output : Obtain heatmaps of the ten model predictions and save them to ''output/plot/heatmap.png'
