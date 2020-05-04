import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #create df, count/total_evets of 0,1,2 for each disaster case
    dfc = pd.DataFrame(index = [0,1,2])
    col_names =df.columns[4:]
    for i in range(0, len(col_names)):
        dfc[col_names[i]] = df.groupby(col_names[i]).count()['message']/len(df)*100
        
    #create df, count/total_evets of 0,1,2 for each disaster case separately for each genre
    
    #direct  
    df_direct = df[df.genre == 'direct' ]
    df_d  = pd.DataFrame(index = [0,1,2])
    #news      
    df_news = df[df.genre == 'news' ]
    df_n  = pd.DataFrame(index = [0,1,2])
    #social    
    df_social = df[df.genre == 'social' ]
    df_s  = pd.DataFrame(index = [0,1,2])
    
    for i in range(0, len(col_names)):
        df_d[col_names[i]] = df_direct.groupby(col_names[i]).count()['message']/len(df_direct)*100
        df_n[col_names[i]] = df_news.groupby(col_names[i]).count()['message']/len(df_news)*100
        df_s[col_names[i]] = df_social.groupby(col_names[i]).count()['message']/len(df_social)*100
        
    # create visuals

    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
      {
            'data': [
                Bar(
                    x=dfc.iloc[1,:],
                    y=list(dfc.columns),
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of disaster cases',
                'yaxis': {
                    'title': "Disaster name"
                },
                'xaxis': {
                    'title': "Disaster number in %"
                }
            }
        } ,
        
        {
            'data': [
                Bar(name='news',
                    x=df_n.iloc[1,:],
                    y=list(df_n.columns),
                    orientation='h'
                ),
                
                Bar(name='Direct',
                    x=df_d.iloc[1,:],
                    y=list(df_d.columns),
                    orientation='h'
                ),
            
                 Bar(name='Social',
                    x=df_s.iloc[1,:],
                    y=list(df_s.columns),
                    orientation='h'
                )
            ],
            'layout': { 
                'title': 'Distribution of disaster cases by genre',
                'yaxis': {
                    'title': "Disaster name"
                },
                'xaxis': {
                    'title': "Disaster number in %"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
