# movie_recommendation_system


This objective of this recommendation system is to create a recommendation engine that takes in a user ID and outputs movies that they would like. 


### Data
There are two files - movies and ratings.
Movies - Movie Id, Title and Genre
Ratings - User Id, Movie Id, Rating and Timestamp.

### Pages

1. Data Overview - summary statistics of the data
2. Train Test Split Overview - the training and testing data is split evenly on the basis of user IDs there ratings. i.e for each user id there are an equal number of ratings in both train and test data.
3. Recommendation Abstract - The recommendation principle used here is collaborative filtering, and the algorithm is the KNNBasic algorithm.
4. Recommendation Demo - Enter a user id and the page will display the following :
   1.Movies from the training data set for that user and how that user rated them
   2.The top 5 movie recommendations and predicted rating for that user
   3. Performance Analysis for that user. - In this the average error in rating is displayed. You will be prompted to enter a threshold for     a good movie (default value 3). This threshold is used to determine the number of true positives, false positives, false negatives and 
    false positives. Movies rated above the threshold are considered "good recommendations", and movies below are considered "bad   
    recommendations". Using the predicted and actual ratings and this threshold, we calculate the metrics. eg: If the movie was rated 2, and 
    the predicted rating is 4 and the threshold was 3, we would count that as a false positive.
    This page also has precision ,recall, f1 score, and the confusion matrix.
   



