
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


class NCF_Data(object):
	"""
	Construct Dataset for NCF
	"""
	def __init__(self, args, ratings):
		self.ratings = ratings
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size

		self.preprocess_ratings = self._reindex(self.ratings)

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
		self.negatives = self._negative_sampling(self.preprocess_ratings)
		random.seed(args.seed)
	
	def _reindex(self, ratings):
		"""
		Process dataset to reindex userID and itemID, also set rating as binary feedback
		"""
		user_list = list(ratings['user_id'].drop_duplicates())
		user2id = {w: i for i, w in enumerate(user_list)}

		item_list = list(ratings['item_id'].drop_duplicates())
		item2id = {w: i for i, w in enumerate(item_list)}

		ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
		ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
		ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
		return ratings

	def _leave_one_out(self, ratings):
		"""
		leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
		"""
		ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
		test = ratings.loc[ratings['rank_latest'] == 1]
		train = ratings.loc[ratings['rank_latest'] > 1]
		assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
		return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

	def _negative_sampling(self, ratings):
		interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
		interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))
		return interact_status[['user_id', 'negative_items', 'negative_samples']]

	def get_train_instance(self):
		users, items, ratings = [], [], []
		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
		for row in train_ratings.itertuples():
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in range(self.num_ng):
				users.append(int(row.user_id))
				items.append(int(row.negatives[i]))
				ratings.append(float(0))  # negative samples get 0 rating
		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

	def get_test_instance(self):
		users, items, ratings = [], [], []
		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
		for row in test_ratings.itertuples():
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in getattr(row, 'negative_samples'):
				users.append(int(row.user_id))
				items.append(int(i))
				ratings.append(float(0))
		dataset = Rating_Datset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=2)



def diversity(pred_items):
	sim = 0
	count = 0

	for i in range(len(pred_items)):

			for j in range(i + 1, len(pred_items) - 1):

					a = pred_items[i]
					b = pred_items[j]

					sim += cosine_sim_df.iloc[a,b]
					count += 1

	return sim/count




def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG, sim = [], [], []

	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()
		#print("this is recs", recommends)

		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))
		sim.append(diversity(recommends))

	return np.mean(HR), np.mean(NDCG), (np.mean(sim))
 
# def pred(model, test_loader, top_k, device):
# 	top_recs_by_user = {}
# 	for user, item, label in test_loader:
# 		user = user.to(device)
# 		item = item.to(device)
# 		label = label.to(device)

#     # Get top recommendations for each user
# 		user_recs = model.top_recommendations(user, args.top_k, device)

#     # Store recommendations in dictionary
# 		top_recs_by_user[user] = user_recs

# # Select a few user_ids to print recommendations for
# 	selected_user_ids = [123, 456, 789]

# # Print recommendations for each selected user
# 	for user_id in selected_user_ids:
# 			if user_id in top_recs_by_user:
# 					user_recs = top_recs_by_user[user_id]
# 					print("Top 10 recommendations for user {}: {}".format(user_id, user_recs))
# 			else:
# 					print("No recommendations found for user {}".format(user_id))


class Generalized_Matrix_Factorization(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.affine_output = nn.Linear(in_features=self.factor_num, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass



class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Multi_Layer_Perceptron, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num
        self.layers = args.layers

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
            # vector = nn.BatchNorm1d()(vector)
            # vector = nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass




class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp =  int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()


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
	default=2,  
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



movies = pd.read_csv('./movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
movies.columns = ['item_id', 'title', 'genres']



from itertools import combinations
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4)
                                             for c in combinations(s.split('|'), r=i)))
tfidf_matrix = tf.fit_transform(movies['genres'])
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])


#from google.colab import files
args = parser.parse_args("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# seed for Reproducibility
#seed_everything(args.seed)

# load data
ml_1m = pd.read_csv(
	'ratings.dat', 
	sep="::", 
	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
	engine='python')


new_ratings = pd.read_csv('./newRatings.csv', sep=',', header=None, engine='python')
ml_1m = pd.concat([ml_1m, new_ratings], ignore_index=True)

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

# construct the train and test datasets
data = NCF_Data(args, ml_1m)
train_loader = data.get_train_instance()
test_loader = data.get_test_instance()
test_loader = random.sample(list(test_loader),1000)

# set model and loss, optimizer
model = NeuMF(args, num_users, num_items)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train, evaluation
best_hr = 0
for epoch in range(1, args.epochs+1):
	model.train() # Enable dropout (if have).
	start_time = time.time()

	for user, item, label in train_loader:
		user = user.to(device)
		item = item.to(device)
		label = label.to(device)

		optimizer.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		writer.add_scalar('loss/Train_loss', loss.item(), epoch)

	model.eval()
	HR, NDCG,sim = metrics(model, test_loader, args.top_k, device)
	#preds = pred(model, test_loader, args.top_k, device)
	writer.add_scalar('Perfomance/HR@10', HR, epoch)
	writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}\tsim: {:.3f} ".format(np.mean(HR), np.mean(NDCG), 1-np.mean(sim)))
 





checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

torch.save(model, "./NCF_trained.pth")

ml_1m.to_csv("./data/ratings.csv")

new_ratings = new_ratings[0:0]

new_ratings.to_csv("./data/newRatings.csv")

# 	if HR > best_hr:
# 		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
# 		if args.out:
# 			torch.save(model.state_dict(), 'checkpoint.pth')

# # download checkpoint file
# 			files.download('checkpoint.pth')

			# if not os.path.exists(MODEL_PATH):
			# 	os.mkdir(MODEL_PATH)
			# torch.save(model, 
			# 	'{}{}.pth'.format(MODEL_PATH, MODEL))

	

writer.close()


















# import os
# import time
# import random
# import argparse
# import numpy as np 
# import pandas as pd 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# #from tensorboardX import SummaryWriter

# import model
# import data_file
# #import eval





# parser = argparse.ArgumentParser()
# parser.add_argument("--seed", 
# 	type=int, 
# 	default=42, 
# 	help="Seed")
# parser.add_argument("--lr", 
# 	type=float, 
# 	default=0.001, 
# 	help="learning rate")
# parser.add_argument("--dropout", 
# 	type=float,
# 	default=0.2,  
# 	help="dropout rate")
# parser.add_argument("--batch_size", 
# 	type=int, 
# 	default=256, 
# 	help="batch size for training")
# parser.add_argument("--epochs", 
# 	type=int,
# 	default=1,  
# 	help="training epoches")
# parser.add_argument("--top_k", 
# 	type=int, 
# 	default=10, 
# 	help="compute metrics@top_k")
# parser.add_argument("--factor_num", 
# 	type=int,
# 	default=32, 
# 	help="predictive factors numbers in the model")
# parser.add_argument("--layers",
#     nargs='+', 
#     default=[64,32,16,8],
#     help="MLP layers. Note that the first layer is the concatenation of user \
#     and item embeddings. So layers[0]/2 is the embedding size.")
# parser.add_argument("--num_ng", 
# 	type=int,
# 	default=4, 
# 	help="Number of negative samples for training set")
# parser.add_argument("--num_ng_test", 
# 	type=int,
# 	default=100, 
# 	help="Number of negative samples for test set")
# parser.add_argument("--out", 
# 	default=True,
# 	help="save model or not")




# args = parser.parse_args("")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #writer = SummaryWriter()

# # seed for Reproducibility
# #seed_everything(args.seed)

# # load data
# ml_1m = pd.read_csv(
# 	'./ratings.dat', 
# 	sep="::", 
# 	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
# 	engine='python')

# new_ratings = pd.read_csv('./newRatings.csv', sep=',', header=None, engine='python')
# ml_1m = pd.concat([ml_1m, new_ratings], ignore_index=True)

# # set the num_users, items
# num_users = ml_1m['user_id'].nunique()+1
# num_items = ml_1m['item_id'].nunique()+1

# # construct the train and test datasets
# data = data_file.NCF_Data(args, ml_1m)
# train_loader = data.get_train_instance()
# test_loader = data.get_test_instance()

# # set model and loss, optimizer
# model = model.NeuMF(args, num_users, num_items)
# model = model.to(device)
# loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)







# # train, evaluation
# best_hr = 0
# for epoch in range(1, args.epochs+1):
# 	model.train() # Enable dropout (if have).
# 	start_time = time.time()

# 	for user, item, label in train_loader:
# 		user = user.to(device)
# 		item = item.to(device)
# 		label = label.to(device)

# 		optimizer.zero_grad()
# 		prediction = model(user, item)
# 		loss = loss_function(prediction, label)
# 		loss.backward()
# 		optimizer.step()
# 		#writer.add_scalar('loss/Train_loss', loss.item(), epoch)

# 	model.eval()
# 	HR, NDCG = eval.metrics(model, test_loader, args.top_k, device)
# 	#preds = pred(model, test_loader, args.top_k, device)
# 	#writer.add_scalar('Perfomance/HR@10', HR, epoch)
# 	#writer.add_scalar('Perfomance/NDCG@10', NDCG, epoch)

# 	elapsed_time = time.time() - start_time
# 	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
# 			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
# 	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    
    
# checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

# torch.save(model, "./NCF_trained.pth")

# ml_1m.to_csv("./data/ml_1m.csv")

# new_ratings = new_ratings[0:0]

# new_ratings.to_csv("./data/new_ratings.csv")

# ratings.to_csv("./data/ratings.csv")

# new_ratings = new_ratings[0:0]

# new_ratings.to_csv("./data/new_ratings.csv")

# 	if HR > best_hr:
# 		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
# 		if args.out:
# 			torch.save(model.state_dict(), 'checkpoint.pth')

# # download checkpoint file
# 			files.download('checkpoint.pth')

			# if not os.path.exists(MODEL_PATH):
			# 	os.mkdir(MODEL_PATH)
			# torch.save(model, 
			# 	'{}{}.pth'.format(MODEL_PATH, MODEL))

	

# writer.close()
