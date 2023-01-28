import pandas as pd
import warnings
import numpy as np

# Ignore DeprecationWarning and UserWarning
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

print('Loading....')

# Load the movies and ratings data
# movies = pd.read_csv('./movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python',encoding='latin-1')
# ratings = pd.read_csv('./ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
# Merge the movies and ratings dataframes on the movieId field
movies = pd.read_csv('./movies.dat', sep='::', header=None, engine='python', encoding='latin-1')


movies.columns = ['movie_id', 'title', 'genres']




def reindex_movies(movies):
    # Extract a list of unique movie IDs from the rating dataset
    movie_list = list(movies['movie_id'].drop_duplicates())
    # Create a dictionary that maps the original movie IDs to the new reindexed IDs
    movie2id = {w: i for i, w in enumerate(movie_list)}
    # Replace the original movie IDs in the movies dataset with the new reindexed IDs
    movies['movie_id'] = movies['movie_id'].apply(lambda x: movie2id[x])
    # Return the reindexed movies dataset
    return movies
movies = reindex_movies(movies)


originalRatings = pd.read_csv('./ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

def orginalReindex(originalRatings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(originalRatings['user_id'].drop_duplicates())
		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(originalRatings['movie_id'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		originalRatings['user_id'] = originalRatings['user_id'].apply(lambda x: user2id[x])
		originalRatings['movie_id'] = originalRatings['movie_id'].apply(lambda x: item2id[x])
		#originalRatings['rating'] = originalRatings['rating'].apply(lambda x: float(x > 0))
		return originalRatings

originalRatings = orginalReindex(originalRatings)









# Create a mapping from genre to number


# Print the genres and their corresponding numbers


# Run the loop until the user inputs 'x'




# Initialize the list
# items = ["item 1", "item 2", "item 3"]

# # Print the instructions and the initial list
# print("instruction 1")
# print("instruction 2")
# print("instruction 3")
# print("\nHere is a list:")
# print(*items, sep='\n')

# # Wait for the user to press a button
# input("\nPress enter to update the list: ")

# # Move the cursor up to the beginning of the list
# print("\033[F" * (len(items) + 1))

# # Update the list
# items = ["item 4", "item 5", "item 6"]

# # Overwrite the initial list with the updated list
# print("\nHere is a list:")
# print(*items, sep='\n', end='\r')





# '\033[1A\033[K'

# print(REMOVE_PREV*8)

# import sys

# # Print 5 lines
# for i in range(5):
#     print(f"Line {i+1}")

# # Wait for the user to press the x key
# while True:
#     c = sys.stdin.read(1)
#     if c == 'x':
#         break

# # Clear the screen
# print("\033c", end="")


# print("This is a line of output")
# print('\033[1A\033[K', end='')import sys

# Print 3 lines
# for i in range(3):
#     print(f"Line {i+1}")

# # Wait for the user to press a key
# input("Press enter to delete the lines: ")

# # Delete the lin
# # es
# for i in range(4):
#     print('\033[1A\033[K', end='')





from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4)
                                             for c in combinations(s.split('|'), r=i)))
tfidf_matrix = tf.fit_transform(movies['genres'])


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])#column wil be our vector
#cosine_sim_df.head()

def genre_recommendations(i, M, items, k=10):
    print(f'looking for items similar to', i)
    """
    Recommends movies based on a similarity dataframe

    Parameters
    ----------
    i : str
        Movie (index of the similarity dataframe)
    M : pd.DataFrame
        Similarity dataframe, symmetric, with movies as indices and columns
    items : pd.DataFrame
        Contains both the title and some other features used to define similarity
    k : int
        Amount of recommendations to return

    """
    ix = M.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))
    closest = M.columns[ix[-1:-(k+2):-1]]
    closest = closest.drop(i, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

#print(genre_recommendations('2001: A Space Odyssey (1968)', cosine_sim_df, movies[['title', 'genres']]))
print('does this get here idk')
userRatings = pd.read_csv('./newRatings.csv', sep=',', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
userRatings = pd.merge(userRatings, movies, on='movie_id')
# title = userRatings.sample().loc['title']
#random_index = np.random.randint(0, len(userRatings))

# extract the value within the column "title"
#title = df.at[random_index, 'title']
rows = userRatings.loc[userRatings['user_id'] == 6040]
random_row = rows.sample()
print(random_row)
title = random_row.iloc[0]['title']

print(f'These recommendations are based on a movie you have watched: {title}')
print(genre_recommendations(title, cosine_sim_df, movies[['title', 'genres']]))