import pandas as np

import os
import time
import random
import argparse
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

originalRatings = pd.read_csv('./ratings.dat', sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

def originalReindex(originalRatings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(originalRatings['user_id'].drop_duplicates())
		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(originalRatings['item_id'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		originalRatings['user_id'] = originalRatings['user_id'].apply(lambda x: user2id[x])
		originalRatings['item_id'] = originalRatings['item_id'].apply(lambda x: item2id[x])
		#originalRatings['rating'] = originalRatings['rating'].apply(lambda x: float(x > 0))
		return originalRatings

originalRatings = originalReindex(originalRatings)

originalRatings.to_csv("ratingsClean.csv", index=False)



movies = pd.read_csv('./movies.dat', sep='::', header=None, engine='python', encoding='latin-1')


movies.columns = ['item_id', 'title', 'genres']




def reindex_movies(movies):
    # Extract a list of unique movie IDs from the rating dataset
    movie_list = list(movies['item_id'].drop_duplicates())
    # Create a dictionary that maps the original movie IDs to the new reindexed IDs
    movie2id = {w: i for i, w in enumerate(movie_list)}
    # Replace the original movie IDs in the movies dataset with the new reindexed IDs
    movies['item_id'] = movies['item_id'].apply(lambda x: movie2id[x])
    # Return the reindexed movies dataset
    return movies
movies = reindex_movies(movies)

movies.to_csv("moviesClean.csv", index=False)