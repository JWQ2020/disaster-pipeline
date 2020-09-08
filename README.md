# disaster-pipeline

## Background
An important application of machine learning is analyse and classify text messages. For this project, the task was to analyse real messages sent by people during disasters, and to classify the message in appropriate categories. The organisation, Figure Eight, have provided a data set containing real-world message data. The aim of the project is to create a machine learning pipeline, and train a model, that is able to categorize such messages, so that they can be sent to an appropriate disaster relief agency. 

## Key components
The three key components of this project are 
1. A code (process_data.py) that extracts and processes the two data files supplied by Figure Eight, that contain message data and message categories.
2. A code (train_classifier.py) that uses the processed data produced by process_data.py, to train the model for categorizing messaegs.
3. A Flask app (run.py) which uses the trained model as a back-end to a web app that enables someone to manually input a message, and obtain a classification of it. It also produces visualisations of the data. 

## Other files
Two Jupyter notebooks have been included, ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb, which were used to develop process_data.py and train_classifier.py respectively. 

## Instructions
Perform the following in this order:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
 
3. View the web app by
  a) navigating to the address http://0.0.0.0:3001/, if running on a local machine, or
  b) https://view########-####.udacity-student-workspaces.com/ where "########-####" should be replaced by the workspace-id, when using a Udacity ID.

4. Enter the message you wish to classify in the text box, and click on the Classify Message button.
