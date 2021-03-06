import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Get the dataset

columns_name = ["user_id","item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep ='\t', names = columns_name) #seperator is /t - tab

movies_title = pd.read_csv("ml-100k/u.item", sep ='\|', header = None , encoding = "ISO-8859-1")

movies_title = movies_title[[0,1]]
movies_title.columns = ["item_id", "title"]
df = pd.merge(df, movies_title, on="item_id")

# Exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

df.groupby('title').mean()['rating'].sort_values(ascending=False)
df.groupby('title').count()['rating'].sort_values(ascending=False)

ratings = pd.DataFrame(df.groupby('title').mean()['rating'])

ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'], bins = 70)

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
#plt.show()

# Creating Movie Recommendation 

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

starwars_user_ratings = moviemat['Star Wars (1977)']
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])

corr_starwars.dropna(inplace=True) # Remove the NaN values

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False)

# prediction Function

def predict_movies(movie_name):
	movie_user_ratings = moviemat[movie_name]
	similar_to_movie = moviemat.corrwith(movie_user_ratings)

	corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
	corr_movie.dropna(inplace=True)

	corr_movie = corr_movie.join(ratings['num of ratings'])
	predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation', ascending=False)

	return predictions

predictions = predict_movies("Titanic (1997)")
print(predictions.head())