from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import streamlit as st
import inspect
import seaborn as sns


st.set_option('deprecation.showPyplotGlobalUse', False)


movies_data =  'https://raw.githubusercontent.com/tarunisunder/movie_recommendation_system/main/movies.csv'
ratings_data =  'https://raw.githubusercontent.com/tarunisunder/movie_recommendation_system/main/ratings.csv'


movies = pd.read_csv(movies_data)
ratings = pd.read_csv(ratings_data)


def data_overview():

  st.title('Movie Recommendation System')
  st.title('Data Overview')
  st.header('Movies Data')
  st.dataframe(movies.head())
  st.header('Ratings Data')
  st.dataframe(ratings.head())

  n_ratings = len(ratings)
  n_movies = len(ratings['movieId'].unique())
  n_users = len(ratings['userId'].unique())

  st.text(f"Number of ratings: {n_ratings}")
  st.text(f"Number of movies: {n_movies}")
  st.text(f"Number of users: {n_users}")
  st.text(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
  st.text(f"Average ratings per movie: {round(n_ratings/n_movies,2)}")

  movie_rating = ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})
  # display(movie_rating.head())
  #make it flat
  movie_rating = movie_rating['rating']
  movie_rating = movie_rating[movie_rating['size'] > 50]
  top_movies = movie_rating.sort_values(by='mean', ascending=False).head()

  #merge the movie_rating with movies
  movie_rating = pd.merge(movie_rating, movies, on='movieId')


  #rename mean to average_rating and size to num_ratings
  movie_rating.rename(columns={'mean': 'average_rating', 'size': 'num_ratings'}, inplace=True)
  st.dataframe(movie_rating.head())

  plt.figure(figsize=(10, 4))
  # sns.histplot(ratings['rating'].tolist(), bins=10)
  plt.hist(ratings['rating'], bins=10)
  plt.title('Distribution of Ratings')
  plt.xlabel('Ratings')
  plt.ylabel('Frequency')
  st.pyplot()





def create_train_test():

  total_dict = ratings['userId'].value_counts().to_dict()
  train_dict = {key: value / 2 for key, value in total_dict.items()}

  train_df_list = []
  test_df_list = []

  for (k,v) in train_dict.items() :
    temp_df = ratings[ratings['userId'] == k].iloc[:,:3].head(int(v))
    temp2_df = ratings[ratings['userId'] == k].iloc[:,:3].tail(int(v))
    train_df_list.append(temp_df)
    test_df_list.append(temp2_df)

  train_df = pd.concat(train_df_list, ignore_index=True)
  test_df = pd.concat(test_df_list, ignore_index=True)


  reader = Reader(rating_scale=(1, 5))

  # Load data into a surprise Dataset
  data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)

  # Build the full trainset
  train_set = data.build_full_trainset()

  test_set = []

  for index, row in test_df.iterrows():
      user_id = row['userId']
      movie_id = row['movieId']
      rating = row['rating']
      test_set.append((user_id, movie_id, rating))

  return train_set, test_set,train_df, test_df



def display_code_widget():
  st.title('Train Test Split Code')
  function_code = inspect.getsource(create_train_test)
  st.code(function_code, language='python')


def recommendation_abstract():
  st.title('Recommendation Abstract')

  st.write("Collaborative Filtering with KNNBasic in a movie recommendation system involves finding similar users or items based on historical preferences. The algorithm uses a K-Nearest Neighbors approach to identify the K most similar users to the target user. Cosine similarity quantifies the resemblance between users. The algorithm then predicts a user's rating for a movie based on the ratings of their similar users and recommends movies with the highest predicted ratings. KNNBasic is part of the Surprise library and can operate in both user-based and item-based collaborative filtering modes.")


def error_analysis(test_df,user_predicted_list,selected_user_id,threshold):
  user_predicted_df =pd.DataFrame(user_predicted_list, columns=['movieId', 'predicted_rating'])
  user_actual_rating = test_df[(test_df['userId'] == selected_user_id)  ]
  user_actual_rating = pd.merge(user_actual_rating, user_predicted_df , on =['movieId'])
  user_actual_rating['diff'] = abs(user_actual_rating['rating'] - user_actual_rating['predicted_rating'])
  avg_error = round(user_actual_rating['diff'].sum() / len(user_actual_rating),2)

  user_actual_rating['ratingBool'] = (user_actual_rating['rating'] > threshold).astype(int)
  user_actual_rating['predictedBool'] = (user_actual_rating['predicted_rating'] > threshold).astype(int)

  actual =user_actual_rating['ratingBool']
  predicted = user_actual_rating['predictedBool']

  conf_matrix = confusion_matrix(user_actual_rating['ratingBool'] , user_actual_rating['predictedBool'] , labels = [False , True])


  precision = round(precision_score(actual, predicted),2)
  recall = round(recall_score(actual, predicted),2)
  f1 = round(f1_score(actual, predicted),2)

  return avg_error, conf_matrix, precision, recall, f1


def feature_enchancements():
  st.title('Feature Enhancements')
  st.write("The principle used to recommend movies is collaborative filtering. As such, there are no explicit features created. The algorithm works by creating a user similarity matrix, and then recommending movies that similar users watched. User similarity is determined by the similarity in their movie ratings. ")



def recommendation_main():

    train_set, test_set, train_df, test_df = create_train_test()
    min_user_id = min(train_df['userId'])
    max_user_id = max(train_df['userId'])
    selected_user_id = st.number_input("Enter user_id:", min_value=min_user_id, max_value=max_user_id, value=30, step=1)


    selected_user_movies = train_df[train_df['userId'] == selected_user_id][['movieId' ,'rating']]

    selected_user_movies = pd.merge(selected_user_movies , movies , on = 'movieId')
    st.header('Movies Watched by this User from Train Set')
    st.dataframe(selected_user_movies)

    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(sim_options=sim_options)

    model.fit(train_set)


    predictions = model.test(test_set)

    predicted_list = []
    for prediction in predictions:
        actual_rating = prediction.r_ui
        predicted_rating = prediction.est
        user_id = prediction.uid
        movie_id = prediction.iid
        predicted_list.append((user_id,movie_id,round(predicted_rating,2)))

    user_predicted_list = [(movie_id,predicted_rating) for (user_id,movie_id,predicted_rating) in predicted_list if user_id == selected_user_id]
    user_predicted_list = sorted(user_predicted_list, key=lambda x: x[1],reverse = True)
    #print(user_predicted_list[0:5])

    best_movies = user_predicted_list[0:5]
    movie_ids = [up[0] for up in best_movies]
    st.header(f'Top 5 Recommended Movies for user {selected_user_id}')
    idx = 1
    for movie_id,pred_rating in (best_movies) :
      movie_info = movies.loc[movies['movieId'] == movie_id][['title']].values
      #pred_rating = round(pred_rating,2)
      st.markdown(f"**{idx}. Movie Name: {movie_info[0][0]}, Predicted Rating: {pred_rating}**")
      idx+=1

    st.header(f'Performance Analysis for user {selected_user_id}')

    threshold = st.number_input("Set the rating threshold for a good movie:", min_value=1, max_value=5, value=3, step=1)

    avg_error, conf_matrix, precision, recall, f1 = error_analysis(test_df,user_predicted_list,selected_user_id,threshold)

    st.markdown(f"**The average error is {avg_error}**")

    st.markdown(f"**The precision is {precision}**")
    st.markdown(f"**The recall is {recall}**")
    st.markdown(f"**The f1 score is {f1}**")
    plt.figure(figsize=(4, 2))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot()

def authenticate(username, password):
    # Hardcoded username and password for simplicity (replace with a secure authentication method)
    valid_username = "user"
    valid_password = "password"
    return username == valid_username and password == valid_password

def login():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Welcome")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        login_button = st.button("Login")

        if login_button:
            if authenticate(username, password):
                st.success("Login successful!")
                st.session_state.authenticated = True
            else:
                st.error("Invalid username or password. Please try again.")
  

def main() :
  login()
  if st.session_state.authenticated:
      page = st.sidebar.selectbox("Select a page", ["Data Overview", "Creating Train and Test sets", "Feature Enhancements", "Recommendation Abstract", "Recommendation Demo"])
      if page == "Data Overview":
          data_overview()
  
      elif page =='Creating Train and Test sets' :
          display_code_widget()

      elif page== "Feature Enhancements" :
          feature_enchancements()
          
      elif page == 'Recommendation Abstract':
          recommendation_abstract()
  
      elif page == "Recommendation Demo" :
          recommendation_main()


if __name__ == '__main__':
    main()



