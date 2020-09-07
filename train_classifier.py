

import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import re
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    """
    Load combined Messages and Categories data file
    Return X (dataframe of features), Y (dataframe of labels)
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Load test to tokenize and clean, including removing url addresses 
    Return list of tokens
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)




def build_model():
    
    pipeline = Pipeline([ 
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {
            'clf__estimator__n_estimators':[150, 200, 250],
            'clf__estimator__max_depth':[5, 10, 12],
            'clf__estimator__min_samples_split': [2, 4],
            'features__text_pipeline__vect__max_features': (None, 5000)
        }
    
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model F1 score, precision and recall for each output category of
    dataset
    """
    y_pred = model.predict(X_test)
    
    columns = y_test.columns
    for i in range(len(columns)):
        print("Category", i, columns[i])
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
    
    pass


def save_model(model, model_filepath):
    """
    Save model as a pickle file 
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()