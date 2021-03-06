'''
ETL for given data.

Loading data from csv files, cleaning and saving in database
'''
import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    
    '''
    load data from files:
    the function takes two parameters, as we have two data files and returns merged dataframe
    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df =  messages.merge(categories, how='outer',\
                               on=['id'])
    return df

def clean_data(df):
    
    '''
    Clean the data:
    the function merged dataframe dataframe as parameter and returns cleand data, 
    which have original massages, genre and 36 individual category columns with corresponding values
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    categories.head()
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # applying a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:len(x)-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column].str[-1:])
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df[category_colnames] = categories
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Load cleaned data in to database:
    the function has two parametersL one cleand dataframe and second database name
    
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse')


if __name__ == '__main__':
    main()