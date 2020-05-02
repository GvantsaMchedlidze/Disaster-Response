import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import numpy as np
import pickle
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    print (database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df.message
    y = df.iloc[:, 4:]
    
    return X, y, df.columns[4:]


def tokenize(text):
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

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2), max_features =5000, tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators= 100, min_samples_split= 2)))
        #('clf',MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.25, 0.15 ),
        #'features__text_pipeline__vect__max_features': (10000, 5000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 100],
        #'clf__estimator__min_samples_split': [2, 4],
        #'clf__estimator__n_neighbors': [5, 2],
        #'clf__estimator__algorithm': ['auto',  'kd_tree', 'brute'],
        'features__transformer_weights': (
            {'features__text_pipeline': 1, 'features__starting_verb': 0.5},
            #{'features__text_pipeline': 0.5, 'features__starting_verb': 1},
            #{'features__text_pipeline': 0.8, 'features__starting_verb': 1}
        )
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters,cv=2,verbose=4)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
        
    y_pred = model.predict(X_test)
    
    for i in range (0,36):
        print (category_names[i],"\n ---------------------------------")
        labels = np.unique(y_pred[:,i])
        confusion_mat = confusion_matrix(Y_test.iloc[:,i], y_pred[:,i], labels=labels)
        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print ("Accuracy:",  (y_pred[:,i] == Y_test.iloc[:,i]).mean(),"\n ---------------------------------")
    
    
        

def save_model(model, model_filepath):
    #with open(model_filepath, "wb") as clf_outfile:
    pickle.dump(model, open(model_filepath, 'wb'))
      


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        model = model.best_estimator_
        print (model) 
       
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