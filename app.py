import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__)

df_tmbd = pd.read_csv('./data/tmdb.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_tmbd['engineered_feature']) # field composed of: director names, cast, keywords, & genre attributes

df_tmbd = df_tmbd.reset_index()
indices = pd.Series(df_tmbd.index, index=df_tmbd['title'])
all_titles = [df_tmbd['title'][i] for i in range(len(df_tmbd['title']))]

def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]
    titles = df_tmbd['title'].iloc[movie_indices]
    dates = df_tmbd['release_date'].iloc[movie_indices]
    df_similar_movies = pd.DataFrame(columns=['Title','Year'])
    df_similar_movies['Title'] = titles
    df_similar_movies['Year'] = dates
    return df_similar_movies

# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        movie_name = flask.request.form['movie_name']
        movie_name = movie_name.title()
        similar_names = difflib.get_close_matches(movie_name,all_titles,cutoff=0.65,n=5)
        if movie_name not in all_titles:
            return(flask.render_template('negative.html',list_similar=similar_names))
        else:
            df_similar_movies = get_recommendations(movie_name)
            names = []
            dates = []
            for i in range(len(df_similar_movies)):
                names.append(df_similar_movies.iloc[i][0])
                dates.append(df_similar_movies.iloc[i][1])

            return flask.render_template('positive.html',movie_names=names,movie_date=dates,search_name=movie_name)

if __name__ == '__main__':
    app.run()
	
#this project is completed following the guide of: https://medium.com/analytics-vidhya/movie-recommendation-system-python-flask-web-application-heroku-deployment-7e39492b640c