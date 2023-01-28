

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
#from tensorboardX import SummaryWriter


print("passed the imports")

class Rating_Datset(torch.utils.data.Dataset):
	def __init__(self, user_list, item_list, rating_list):
		super(Rating_Datset, self).__init__()
		self.user_list = user_list
		self.item_list = item_list
		self.rating_list = rating_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):
		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		
		return (
			torch.tensor(user, dtype=torch.long),
			torch.tensor(item, dtype=torch.long),
			torch.tensor(rating, dtype=torch.float)
			)


# class NCF_Data(object):
# 	"""
# 	Construct Dataset for NCF
# 	"""
# 	def __init__(self, args, ratings):
# 		self.ratings = ratings
# 		self.num_ng = args.num_ng
# 		self.num_ng_test = args.num_ng_test
# 		self.batch_size = args.batch_size

# 		self.preprocess_ratings = self._reindex(self.ratings)

# 		self.user_pool = set(self.ratings['user_id'].unique())
# 		self.item_pool = set(self.ratings['item_id'].unique())

# 		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
# 		self.negatives = self._negative_sampling(self.preprocess_ratings)
# 		random.seed(args.seed)
	
# 	def _reindex(self, ratings):
# 		"""
# 		Process dataset to reindex userID and itemID, also set rating as binary feedback
# 		"""
# 		user_list = list(ratings['user_id'].drop_duplicates())
# 		user2id = {w: i for i, w in enumerate(user_list)}

# 		item_list = list(ratings['item_id'].drop_duplicates())
# 		item2id = {w: i for i, w in enumerate(item_list)}

# 		ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
# 		ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
# 		ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
# 		return ratings

# 	def _leave_one_out(self, ratings):
# 		"""
# 		leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
# 		"""
# 		ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
# 		test = ratings.loc[ratings['rank_latest'] == 1]
# 		train = ratings.loc[ratings['rank_latest'] > 1]
# 		assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
# 		return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

# 	def _negative_sampling(self, ratings):
# 		interact_status = (
# 			ratings.groupby('user_id')['item_id']
# 			.apply(set)
# 			.reset_index()
# 			.rename(columns={'item_id': 'interacted_items'}))
# 		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
# 		interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))
# 		return interact_status[['user_id', 'negative_items', 'negative_samples']]

# 	def get_train_instance(self):
# 		users, items, ratings = [], [], []
# 		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')
# 		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
# 		for row in train_ratings.itertuples():
# 			users.append(int(row.user_id))
# 			items.append(int(row.item_id))
# 			ratings.append(float(row.rating))
# 			for i in range(self.num_ng):
# 				users.append(int(row.user_id))
# 				items.append(int(row.negatives[i]))
# 				ratings.append(float(0))  # negative samples get 0 rating
# 		dataset = Rating_Datset(
# 			user_list=users,
# 			item_list=items,
# 			rating_list=ratings)
# 		return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

# 	def get_test_instance(self):
# 		users, items, ratings = [], [], []
# 		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
# 		for row in test_ratings.itertuples():
# 			users.append(int(row.user_id))
# 			items.append(int(row.item_id))
# 			ratings.append(float(row.rating))
# 			for i in getattr(row, 'negative_samples'):
# 				users.append(int(row.user_id))
# 				items.append(int(i))
# 				ratings.append(float(0))
# 		dataset = Rating_Datset(
# 			user_list=users,
# 			item_list=items,
# 			rating_list=ratings)
# 		return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=2)


# def hit(ng_item, pred_items):
# 	if ng_item in pred_items:
# 		return 1
# 	return 0


# def ndcg(ng_item, pred_items):
# 	if ng_item in pred_items:
# 		index = pred_items.index(ng_item)
# 		return np.reciprocal(np.log2(index+2))
# 	return 0


# def metrics(model, test_loader, top_k, device):
# 	HR, NDCG = [], []

# 	for user, item, label in test_loader:
# 		user = user.to(device)
# 		item = item.to(device)

# 		predictions = model(user, item)
# 		_, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				item, indices).cpu().numpy().tolist()
# 		#print("this is recs", recommends)

# 		ng_item = item[0].item() # leave one-out evaluation has only one item per user
# 		HR.append(hit(ng_item, recommends))
# 		NDCG.append(ndcg(ng_item, recommends))

# 	return np.mean(HR), np.mean(NDCG)


# class Generalized_Matrix_Factorization(nn.Module):
#     def __init__(self, args, num_users, num_items):
#         super(Generalized_Matrix_Factorization, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.factor_num = args.factor_num

#         self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
#         self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

#         self.affine_output = nn.Linear(in_features=self.factor_num, out_features=1)
#         self.logistic = nn.Sigmoid()

#     def forward(self, user_indices, item_indices):
#         user_embedding = self.embedding_user(user_indices)
#         item_embedding = self.embedding_item(item_indices)
#         element_product = torch.mul(user_embedding, item_embedding)
#         logits = self.affine_output(element_product)
#         rating = self.logistic(logits)
#         return rating

#     def init_weight(self):
#         pass


# class Multi_Layer_Perceptron(nn.Module):
#     def __init__(self, args, num_users, num_items):
#         super(Multi_Layer_Perceptron, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.factor_num = args.factor_num
#         self.layers = args.layers

#         self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
#         self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

#         self.fc_layers = nn.ModuleList()
#         for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
#             self.fc_layers.append(nn.Linear(in_size, out_size))

#         self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
#         self.logistic = nn.Sigmoid()

#     def forward(self, user_indices, item_indices):
#         user_embedding = self.embedding_user(user_indices)
#         item_embedding = self.embedding_item(item_indices)
#         vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
#         for idx, _ in enumerate(range(len(self.fc_layers))):
#             vector = self.fc_layers[idx](vector)
#             vector = nn.ReLU()(vector)
#             # vector = nn.BatchNorm1d()(vector)
#             # vector = nn.Dropout(p=0.5)(vector)
#         logits = self.affine_output(vector)
#         rating = self.logistic(logits)
#         return rating

#     def init_weight(self):
#         pass


# class NeuMF(nn.Module):
#     def __init__(self, args, num_users, num_items):
#         super(NeuMF, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.factor_num_mf = args.factor_num
#         self.factor_num_mlp =  int(args.layers[0]/2)
#         self.layers = args.layers
#         self.dropout = args.dropout

#         self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
#         self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

#         self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
#         self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

#         self.fc_layers = nn.ModuleList()
#         for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
#             self.fc_layers.append(torch.nn.Linear(in_size, out_size))
#             self.fc_layers.append(nn.ReLU())

#         self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf, out_features=1)
#         self.logistic = nn.Sigmoid()
#         self.init_weight()

#     def init_weight(self):
#         nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
#         nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
#         nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
#         nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
#         for m in self.fc_layers:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
                
#         nn.init.xavier_uniform_(self.affine_output.weight)

#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, user_indices, item_indices):
#         user_embedding_mlp = self.embedding_user_mlp(user_indices)
#         item_embedding_mlp = self.embedding_item_mlp(item_indices)

#         user_embedding_mf = self.embedding_user_mf(user_indices)
#         item_embedding_mf = self.embedding_item_mf(item_indices)

#         mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
#         mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

#         for idx, _ in enumerate(range(len(self.fc_layers))):
#             mlp_vector = self.fc_layers[idx](mlp_vector)

#         vector = torch.cat([mlp_vector, mf_vector], dim=-1)
#         logits = self.affine_output(vector)
#         rating = self.logistic(logits)
#         return rating.squeeze()


parser = argparse.ArgumentParser()
parser.add_argument("--seed", 
	type=int, 
	default=42, 
	help="Seed")
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.2,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=1,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+', 
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user \
    and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="Number of negative samples for training set")
parser.add_argument("--num_ng_test", 
	type=int,
	default=100, 
	help="Number of negative samples for test set")
parser.add_argument("--out", 
	default=True,
	help="save model or not")

print("past all the loading stuff")


#from google.colab import files
args = parser.parse_args("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter()

# seed for Reproducibility
#seed_everything(args.seed)

# load data
ml_1m = pd.read_csv(
	'./ratings.dat', 
	sep="::", 
	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
	engine='python')

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

# construct the train and test datasets
from data_file import NCF_Data
data = NCF_Data(args, ml_1m)
# train_loader = data.get_train_instance()
# test_loader = data.get_test_instance()
from model import NeuMF
# set model and loss, optimizer
model = NeuMF(args, num_users, num_items)
model = model.to(device)
# loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)




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
		originalRatings['rating'] = originalRatings['rating'].apply(lambda x: float(x > 0))
		return originalRatings

originalRatings = originalReindex(originalRatings)

#####adding the content based stuff#######

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
    #print(f'looking for items similar to', i)
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
#print('does this get here idk')
userRatings = pd.read_csv('./newRatings.csv', sep=',', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
userRatings = pd.merge(userRatings, movies, on='item_id')



######################################








device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if device == torch.device("cuda"):
    checkpoint = torch.load("./NCF_trained.pth")
else:
    checkpoint = torch.load("./NCF_trained.pth", map_location=torch.device("cpu"))



# movies = pd.read_csv("./data/movies.csv")
# ratings = pd.read_csv("./data/ratings.csv")
movies = reindex_movies(movies)
model = NeuMF(args, num_users, num_items).to(device)
checkpoint = torch.load("./NCF_trained.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint)





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
        
        
    
        
                


import datetime



###this class is the basis for the recommender system

##it involves the menu that is used to navigate the sysem

class RecommenderSystems:
    def __init__(self):
        self.user = None
        self.state = 'main_menu'
        self.new = False
        self.added = True
        
        
    def main_menu(self):
        while True:
            if self.state == 'main_menu':
                print("Welcome to our movie recommender system!")
                print(' ')
                print('For this recommender system, we use your previous movie interactions to make personalised recommendations')
                print('All of this information is stored anonymously')
                print(' ')
                print('1: Log in')
                print('2: Create a new account')
                print('3: Exit')
                print(' ')

                
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
                print(' ')
                print(f'Welcome user {self.user}')
                print(' ')

                
                print('1: Display your movie recommendations')
                print('2: Add a new watched movie')
                print('3: Exit')
                print(' ')
                choice = input("Please enter your choice:")
                
                if choice == '1' and self.new ==False:
                    self.chooseGenre()

                elif choice == '1' and self.new ==True:
                    self.randomContent()
                    
                elif choice == '2':
                    self.add_rating(ml_1m, movies)
                    
                elif choice == '3':
                    print("Thank you for using our movie recommender system, we hope to see you again soon!")
                    break
#                     self.exit()
    
                    
                else:
                    print("Invalid choice, please try again!")



    #user is able to log in. They can choose a value between 0 and 6039 as an id, however more values will be added as new users are added               
    def log_in(self):
        
        while True:
            print(' ')
            user_id = int(input("Please enter your user id!"))

            if user_id in ml_1m['user_id'].unique():
                self.user = int(user_id)
                self.state = 'logged_in'
                break

            else:
                print('Invalid user id, please try again!')
                
    #the users are able to add more movies to their profile by choosing a movie from the dataset
    def add_rating(self, ml_1m, movies):
        
        
        #check for duplicates would be a good one
        # print(self.new)
        # print(self.added)
        if self.added == True:
            
            while True:
                #giving them updates on what is happening
                print('The movies you add will not be used in your profile until our model is updated')
                print('Please be patient until this is done!')
                print(' ')
                movie_name = input("Enter the name of the movie you have watched: ")
                movie = movies[movies['title'] == movie_name]
                if movie.empty:
                    print(' ')
                    print(f"Movie '{movie_name}' not found in the dataset. Please enter a valid movie name.")
                    # choices = movies['title'].tolist()
                    # match = process.extractOne(movie_name,choices,scorer=fuzz.token_set_ratio)
                    # print(f'Did you mean {match[0]}?')

                    #perhaps use levenstein trick mentioned here for fuzzy wuzzy matching https://medium.com/geekculture/creating-content-based-movie-recommender-with-python-7f7d1b739c63
                    print("Please select your next option")
                    print("1: Try again")
                    print("2: Exit to the main menu")

    #                 while True:
                    print(' ')
                    choice = input("What would you like to do?")
                    if choice == '1':
                        continue

                    elif choice != '1':
                        break
   
                else:
                    movie_id = movie.iloc[0]['item_id']
    #                 new_rating = pd.DataFrame({'userId': [self.user], 'movieId': [movie_id], 'rating': [1.0]})
    #                 ratings = pd.concat([ml_1m, new_rating])
                    with open("newRatings.csv", "a") as file_object:
                        file_object.write(
                        f"{self.user},{movie_id},1.0,{int(datetime.datetime.now().timestamp())}\n"
                )




                    print(f"Movie '{movie_name}' has been added to your ratings.")
                    print(' ')
                    print("Please select your next option")
                    print("1: Add another movie")
                    print("2: Exit to the main menu")

    #                 while True:
                    print(' ')
                    choice = input("What would you like to do?")
                    if choice == '1':
                        continue

                    elif choice != '1':
                        break

            return ml_1m
        else:


            #this is for when a new user is added to the system
            #they need to add 5 movies before they can continue to use the system
            added = 0
            while added <5:
                movie_name = input("Enter the name of the movie you have watched: ")
                movie = movies[movies['title'] == movie_name]
                if movie.empty:
                    print(f"Movie '{movie_name}' not found in the dataset. Please enter a valid movie name.")
                    # choices = movies['title'].tolist()
                    # match = process.extractOne(movie_name,choices,scorer=fuzz.token_set_ratio)
                    # print(f'Did you mean {match[0]}?')

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
                else:#this adds the new movie wathches to a new dataset, this will later be concatenated with the main ratings dataset and the NeuMF model will be retrained
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
            print(' ')
            print("Taking you back to menu!")
            #self.user = next_user_id
            self.state = 'logged_in'
            self.added = True
            
                
        
            
            
     
        
    #this is the method for choosing content based recommendations. A user has a random movie chosen from those that they have watched
    #recommendations are then made from this, based on the genre of the movie
    def randomContent(self):
        #contains explanations on what is happening with the system
        print('While we wait for our algorithm to update with your account and preferences,')
        print('we will base our recommendations on the genre of films youve watched')
        while True:
            print(' ')
            print('Press 1 to view genre based recommendations from a random film you/ve watched')
            print('Press 2 to go back to the main menu')
            print(' ')

            choice = input("What would you like to do?")
            print(' ')
           
            if choice == '1':
                rows = userRatings.loc[userRatings['user_id'] == self.user]
                random_row = rows.sample()
                #print(random_row)
                title = random_row.iloc[0]['title']


                print(f'These recommendations are based on a movie you have watched: {title}')
                print(' ')
                recs = genre_recommendations(title, cosine_sim_df, movies[['title', 'genres']])
                for i, row in recs.iterrows():
                    print('')
                    print(f"{row['title']}")

            else:
                self.state = 'logged_in'
                break


                    
                                

    #allows the user to choose the genre they want to receive recommendations for
    #this works with the recommendations made by the NeuMF model
    def chooseGenre(self):
        
        ##can use the self.new = False thing here to check if we wanna use content based or collaborative etc, or even implement the 
        
        ##clean up a bit for sure
        print('Here are a list of genres available on our system')
        print('')
        print('Which genre do you want to watch today?')
        print(' ')


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
            print(' ')
            print('Enter a genre number to view your personal recommendations from this genre')
            print('Press x to exit to the main menu')
            print(' ')
        
            genre_number = input('Please select an option')
        #     for i in range(12):
        #         print('\033[1A\033[K', end='')


            if genre_number == 'x':
                self.state = 'logged_in'
                break

            elif genre_number == '18':
                top_recs = getRecs(self, "all")
            #     clear_output()
                print(' ')
                #displaying the recommendations
                print(f'Here are your top 10 movie recommendations:')
                print(' ')
                for i, row in top_recs.iterrows():
                    print('')
                    print(f"{row['title']}")

            else:

                genre_number = int(genre_number)

                # Get the corresponding genre
                genre = number_to_genre[genre_number]

                top_recs = getRecs(self, genre)
            #     clear_output()
                print(' ')
                print(f'Here are your top 10 movie recommendations in {genre}:')
                print(' ')
                for i, row in top_recs.iterrows():
                    print('')
                    print(f"{row['title']}")
        
        
    #this allows new users to create an account
    #it selects the max +1 value user id from the ratings dataset to assign a user id to the new user
    def create_account(self):
        next_user_id = ml_1m['user_id'].max() + 1
        self.user = int(next_user_id)
        self.added = False
        self.new = True
        print(f"Your user ID is {next_user_id}")
        print("Please enter 5 movies that you have watched before you can proceed:")
        print('When you have entered these movies, we will be able to generate recommendations of films that are similar to those that you have watched, based on their genre')
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
        
                
            
                