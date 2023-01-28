print("Loading...")

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
import datetime
#from tensorboardX import SummaryWriter
#pip install fuzzywuzzy
#import fuzzywuzzy
#from fuzzywuzzy import fuzz, process

import train
import data_file
import eval
import model


from train import ml_1m
from train import args

print("imports done")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if device == torch.device("cuda"):
    checkpoint = torch.load("./NCF_trained.pth")
else:
    checkpoint = torch.load("./NCF_trained.pth", map_location=torch.device("cpu"))

num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1


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


# movies = pd.read_csv("./data/movies.csv")
# ratings = pd.read_csv("./data/ratings.csv")
movies = reindex_movies(movies)
model = model.NeuMF(args, num_users, num_items).to(device)
checkpoint = torch.load("./NCF_trained.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint)

print("model loaded")





# Set the model to evaluation mode
model.eval()
user_interacted_items = ml_1m.groupby("user_id")["item_id"].apply(list).to_dict()

all_movieIds = ml_1m["item_id"].unique()


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






# def getRecs(u):
#     interacted_items = user_interacted_items[u]
#     not_interacted_items = list(set(all_movieIds) - set(interacted_items))
#   #print(len(not_interacted_items))
#     user_index_tensor = torch.tensor([u]*len(not_interacted_items)).to("cuda")
#     item_indices = torch.tensor(not_interacted_items).to("cuda")
#     predicted_labels = model(user_index_tensor, item_indices).to("cuda")
#     top10_items = [x for _, x in sorted(zip(predicted_labels, not_interacted_items), key=lambda pair: pair[0], reverse=True)][:10]
#     #print(top10_items)
#     return movies.loc[movies['movieId'].isin(top10_items)]

print("up to the functions")
def getRecs(self, genre):
    
    
    if self.new == False:
        if genre == "all":
            interacted_items = user_interacted_items[self.user]
            not_interacted_items = list(set(all_movieIds) - set(interacted_items))
          #print(len(not_interacted_items))
            user_index_tensor = torch.tensor([self.user]*len(not_interacted_items)).to(device)
            item_indices = torch.tensor(not_interacted_items).to(device)
            predicted_labels = model(user_index_tensor, item_indices).to(device)
            top10_items = [x for _, x in sorted(zip(predicted_labels, not_interacted_items), key=lambda pair: pair[0], reverse=True)][:10]
            #print(top10_items)
            return movies.loc[movies['item_id'].isin(top10_items)]
        else:


            interacted_items = user_interacted_items[self.user]
            not_interacted_items = list(set(all_movieIds) - set(interacted_items))
            # filter the movies dataset by genre
            genre_movies = movies[movies['genres'].str.contains(genre)]
            # get the movieIds of the filtered movies
            genre_movieIds = genre_movies['item_id'].tolist()
            # filter the not_interacted_items by the movieIds of the filtered movies
            not_interacted_items = list(set(genre_movieIds) & set(not_interacted_items))
            user_index_tensor = torch.tensor([self.user]*len(not_interacted_items)).to(device)
            item_indices = torch.tensor(not_interacted_items).to(device)
            predicted_labels = model(user_index_tensor, item_indices).to(device)
            top10_items = [x for _, x in sorted(zip(predicted_labels, not_interacted_items), key=lambda pair: pair[0], reverse=True)][:10]
            return movies.loc[movies['item_id'].isin(top10_items)]
        
        
    else:
        print("This needs content based or random recs from the non personalised system")
#         genre_number = int(genre_number)
    
#         # Get the corresponding genre
#         genre = number_to_genre[genre_number]
        merged = pd.merge(movies, ml_1m, on='item_id')

        # Filter the merged dataframe to only include movies of the selected genre
        genre_movies = merged[merged['genres'].str.contains(genre)]

        # Group the movies by movieId and compute the count and mean of the ratings
        genre_movies = genre_movies.groupby('item_id').agg({'rating': ['count', 'mean']})

        # Compute the weighted average mean rating for each movie
        genre_movies['weighted_mean'] = genre_movies.apply(lambda row: (row[('rating', 'mean')] * row[('rating', 'count')]) / row[('rating', 'count')], axis=1)

        # Sort the movies by weighted mean rating and get the top 10
        top_10 = genre_movies.sort_values(by='weighted_mean', ascending=False).head(10)

        # Merge the top 10 movies with the movies dataframe to get the movie titles
        top_10 = pd.merge(top_10, movies, on='item_id')
        return top_10





#try self state system to make it a bit more easily readable perhaps and allow us to move between the states
class RecommenderSystems:
    def __init__(self):
        self.user = None
        self.state = 'main_menu'
        self.new = False
        self.added = True
        #self.new to check if theyre a new user or not - if so use the new ratings file as part of something perhaps, generate some similar movies to them based on what theyve seen before???
        #minor thing - not absolutely necessary
        
    def main_menu(self):
        while True:
            if self.state == 'main_menu':
                print("Welcome to our movie recommender system!")
                print('1: Log in')
                print('2: Create a new account')
                print('3: Exit')
                choice = input("Please enter your choice:")
                
                if choice == '1':
                    self.log_in()
                    
                elif choice == '2':
                    self.create_account()
                    
                elif choice == '3':
                    #self.exit()
                    print("Thank you for using our movie recommender system, we hope to see you again soon!")
                    break
                    
                else:
                    print("Invalid choice, please try again!")
                    
            elif self.state == 'logged_in':
                
                
                ###
                #do some explanations on what data has been collected and why
                #how are recommendations made - explain perhaps - look at order and ranking
                
                ####
                
                print(f'Welcome  user {self.user}')
                print('1: Display your top movie recommendation for a specific genre')
                print('2: Add a new watched movie')
                print('3: Exit')
                choice = input("Please enter your choice:")
                
                if choice == '1':
                    self.chooseGenre()
                    
                elif choice == '2':
                    self.add_rating(ml_1m, movies)
                    
                elif choice == '3':
                    print("Thank you for using our movie recommender system, we hope to see you again soon!")
                    break
#                     self.exit()
    
                    
                else:
                    print("Invalid choice, please try again!")
                    
    def log_in(self):
        
        while True:
            user_id = int(input("Please enter your user id!"))

            if user_id in ml_1m['user_id'].unique():
                self.user = int(user_id)
                self.state = 'logged_in'
                break

            else:
                print('Invalid user id, please try again!')
                
                
    def add_rating(self, ml_1m, movies):
        
        
        #check for duplicates would be a good one
        print(self.new)
        print(self.added)
        if self.added == True:
            
            while True:
                movie_name = input("Enter the name of the movie you have watched: ")
                movie = movies[movies['title'] == movie_name]
                if movie.empty:
                    print(f"Movie '{movie_name}' not found in the dataset. Please enter a valid movie name.")
                    # choices = movies['title'].tolist()
                    # match = process.extractOne(movie_name,choices,scorer=fuzz.token_set_ratio)
                    # print(f'Did you mean {match[0]}?')

                    #perhaps use levenstein trick mentioned here for fuzzy wuzzy matching https://medium.com/geekculture/creating-content-based-movie-recommender-with-python-7f7d1b739c63
                    print("Please select your next option")
                    print("1: Try again")
                    print("2: Exit to the main menu")

    #                 while True:

                    choice = input("What would you like to do?")
                    if choice == '1':
                        continue

                    elif choice != '1':
                        break
    #                 go_back = input("Do you want to go back to the main menu? [y/n]")
    #                 if go_back.lower() == 'y':
    #                     break
                else:
                    movie_id = movie.iloc[0]['item_id']
    #                 new_rating = pd.DataFrame({'userId': [self.user], 'movieId': [movie_id], 'rating': [1.0]})
    #                 ratings = pd.concat([ml_1m, new_rating])
                    with open("newRatings.csv", "a") as file_object:
                        file_object.write(
                        f"{self.user},{movie_id},1.0,{int(datetime.datetime.now().timestamp())}\n"
                )




                    print(f"Movie '{movie_name}' has been added to your ratings.")
                    print("Please select your next option")
                    print("1: Add another movie")
                    print("2: Exit to the main menu")

    #                 while True:

                    choice = input("What would you like to do?")
                    if choice == '1':
                        continue

                    elif choice != '1':
                        break

            return ml_1m
        else:
            added = 0
            while added <5:
                movie_name = input("Enter the name of the movie you have watched: ")
                movie = movies[movies['title'] == movie_name]
                if movie.empty:
                    print(f"Movie '{movie_name}' not found in the dataset. Please enter a valid movie name.")
                    choices = movies['title'].tolist()
                    match = process.extractOne(movie_name,choices,scorer=fuzz.token_set_ratio)
                    print(f'Did you mean {match[0]}?')

                    #perhaps use levenstein trick mentioned here for fuzzy wuzzy matching https://medium.com/geekculture/creating-content-based-movie-recommender-with-python-7f7d1b739c63
#                     print("Please select your next option")
#                     print("1: Try again")
#                     print("2: Exit to the main menu")

#     #                 while True:

#                     choice = input("What would you like to do?")
#                     if choice == '1':
#                         continue

#                     elif choice != '1':
#                         break
#     #                 go_back = input("Do you want to go back to the main menu? [y/n]")
#     #                 if go_back.lower() == 'y':
#     #                     break
                else:
                    movie_id = movie.iloc[0]['item_id']
    #                 new_rating = pd.DataFrame({'userId': [self.user], 'movieId': [movie_id], 'rating': [1.0]})
    #                 ratings = pd.concat([ml_1m, new_rating])
                    with open("newRatings.csv", "a") as file_object:
                        file_object.write(
                        f"{self.user},{movie_id},1.0,{int(datetime.datetime.now().timestamp())}\n"
                )




                    print(f"Movie '{movie_name}' has been added to your ratings.")
                    added+=1
            self.added = True
                    
            print("Thank you for adding your movies. You can now move on.")
            print("Taking you back to menu!")
            #self.user = next_user_id
            self.state = 'logged_in'
            self.added = True
            
                
        
            
            
     
        
        
        
    def chooseGenre(self):
        
        ##can use the self.new = False thing here to check if we wanna use content based or collaborative etc, or even implement the 
        
        ##clean up a bit for sure
        print('Here are a list of genres available on our system')
        print('')
        print('Which genre do you want to watch today?')


        ####
        #perhaps check to see if we need to use content or collaborative filtering by checking the number of movies a user has rated
        #can do a switch to use a different method
        ###########
        
        
        # Print the genres and their corresponding numbers
        print('Genres:')
        for genre, number in genre_to_number.items():
            print(f'{number}: {genre}')
        print('18: All')

        # Run the loop until the user inputs 'x'
        #printed = False
        while True:


        #     if printed == False:
        #         print(f'Top 10 movies:')
        #         for _ in range (10):
        #             print("Film:")

        # Prompt the user to enter a genre number
            genre_number = input('Enter a genre number (or "x" to exit): ')
        #     for i in range(12):
        #         print('\033[1A\033[K', end='')


            if genre_number == 'x':
                self.state = 'logged_in'
                break

            elif genre_number == '18':
                top_recs = getRecs(self, "all")
            #     clear_output()
                print(f'Here are your top 10 movie recommendations:')
                for i, row in top_recs.iterrows():
                    print('')
                    print(f"{row['title']}")

            else:

                genre_number = int(genre_number)

                # Get the corresponding genre
                genre = number_to_genre[genre_number]

                top_recs = getRecs(self, genre)
            #     clear_output()
                print(f'Here are your top 10 movie recommendations in {genre}:')
                for i, row in top_recs.iterrows():
                    print('')
                    print(f"{row['title']}")
        
        
            
    def create_account(self):
        next_user_id = ml_1m['user_id'].max() + 1
        self.user = int(next_user_id)
        self.added = False
        self.new = True
        print(f"Your user ID is {next_user_id}")
        print("Please enter 5 movies that you have watched before you can proceed:")
        #because .... we can then generate some recs based on the content of films youve watched before
#         for i in range(5):
        self.add_rating(ml_1m, movies)
#         self.added = True
#         print("Thank you for adding your movies. You can now move on.")
#         print("Taking you back to menu!")
#         self.user = next_user_id
#         self.state = 'logged_in'
        
        
        
        
#     def exit(self):
        
           
        





recommender_system = RecommenderSystems()
recommender_system.main_menu()
        
                
            
                