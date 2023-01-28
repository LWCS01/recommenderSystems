import numpy as np
import torch

from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4)
                                             for c in combinations(s.split('|'), r=i)))
tfidf_matrix = tf.fit_transform(movies['genres'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])#column wil be our vector

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

	return np.mean(HR), np.mean(NDCG), (1-np.mean(sim))




	##diversity 
	from sklearn.metrics.pairwise import cosine_similarity

# # Assume that `tfidf_matrix` is a 2D array containing the TF-IDF vectors of all films in the dataset
# # and `recommended_films` is a list of the ten recommended films

# # Find the indices of the recommended films in the dataset
# indices = [i for i, title in enumerate(all_films) if title in recommended_films]

# # Extract the TF-IDF vectors for the recommended films
# recommended_tfidf = tfidf_matrix[indices]

# # Calculate the pairwise cosine similarities between the recommended films
# similarities = cosine_similarity(recommended_tfidf)

# # Calculate the average similarity
# average_similarity = similarities.mean()

# # The lower the average similarity, the more diverse the recommended films are
# print("Diversity of recommended films:", 1 - average_similarity)



###novelty 
# import numpy as np

# # Assume that `user_item_matrix` is a 2D array representing the user-item matrix
# # and `recommended_films` is a list of the ten recommended films for a sample of users

# # Find the most popular movies among the users
# popularity = user_item_matrix.sum(axis=0)
# most_popular = np.argsort(popularity)[-10:]

# # Find the number of unique recommended films that are not in the most popular movies
# novelty = len(set(recommended_films) - set(most_popular))

# # The higher the number of unique recommended films, the more novel the recommendations are
# print("Novelty of recommendations:", novelty)



#This code finds the most popular movies by summing up the user-item matrix along the rows, then taking the top 10 movies.
# Then it finds the number of unique recommended films that are not in the most popular movies. Finally, it prints the novelty of the recommendations by taking the number of unique recommended films.

# Another approach is to use the concept of popularity bias, which is the measure of how much the recommendations are skewed towards the popular items. You can calculate the popularity bias by comparing the popularity of the recommended items with the overall popularity of all items.

# Keep in mind that this is just one approach to calculating novelty, and there may be other methods that are more appropriate for your specific task.