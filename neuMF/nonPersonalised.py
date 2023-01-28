import pandas as pd
import warnings

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








from IPython.display import clear_output
merged = pd.merge(movies, originalRatings, on='movie_id')

# Create a mapping from genre to number
genres = set()
genre_to_number = {}
number_to_genre = {}
current_number = 0

# Iterate over the rows of the movies dataframe
for _, row in movies.iterrows():
    # Split the genres field into a list
    movie_genres = row['genres'].split('|')
    # Add each genre to the set and mapping
    for genre in movie_genres:
        if genre not in genres:
            genres.add(genre)
            genre_to_number[genre] = current_number
            number_to_genre[current_number] = genre
            current_number += 1


print('Welcome to our movie recommender system')
print(' ')

print('These recommendations are based on the top 100 average weighted movies within each genre')
print('A random sample of 10 movies from these 100 movies are chosen for the recommendations')
print(' ')
print('Here is a list of genres available on our system')
print(' ')

print('Choose a genre to see the 10 most popular movies in the dataset')
print(' ')


# Print the genres and their corresponding numbers
print('Genres:')
for genre, number in genre_to_number.items():
    print(f'{number}: {genre}')

# Run the loop until the user inputs 'x'
#printed = False
while True:
    # print('Genres:')
    # for genre, number in genre_to_number.items():
    #     print(f'{number}: {genre}')


    # if printed == False:
    #     print(f'Top 10 movies:')
    #     for _ in range (10):
    #         print("Film:")
        
    # Prompt the user to enter a genre number
    genre_number = input('Enter a genre number (or "x" to exit): ')
    # for i in range(12):
    #     print('\033[1A\033[K', end='')
        



    
    if genre_number == 'x':
        break
    
    # Convert the input to an integer
    genre_number = int(genre_number)
    
    # Get the corresponding genre
    genre = number_to_genre[genre_number]
    
    # Filter the merged dataframe to only include movies of the selected genre
    genre_movies = merged[merged['genres'].str.contains(genre)]
    
    # Group the movies by movieId and compute the count and mean of the ratings
    genre_movies = genre_movies.groupby('movie_id').agg({'rating': ['count', 'mean']})
    
    # Compute the weighted average mean rating for each movie
    genre_movies['weighted_mean'] = genre_movies.apply(lambda row: (row[('rating', 'mean')] * row[('rating', 'count')]) / row[('rating', 'count')], axis=1)
    
    # Sort the movies by weighted mean rating and get the top 10
    top_10 = genre_movies.sort_values(by='weighted_mean', ascending=False).head(10)
    #print(top_10)
    
    # Merge the top 10 movies with the movies dataframe to get the movie titles
    top_10 = pd.merge(top_10, movies, on='movie_id')
    
    # Print the top 10 movies
    #clear_output()
    print(f'Top 10 {genre} movies:')
    print(' ')
    for _, row in top_10.iterrows():
        print(' ')
        print(row['title'])
    #printed = True



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