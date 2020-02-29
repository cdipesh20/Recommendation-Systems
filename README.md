# Recommendation-Systems
In this project, I have created 3 types of Recommendation Engines on the Yelp Dataset to predict Stars/Ratings for a given User and Business.

__1. Model Based Collaborative Filtering with Spark MLlib__

Trained the Recommendation Model using Alternating Mean Squares and evaluated the Testing Data on this trained model.  
RMSE Achieved: 1.24

__2. User Based Collaborative Filtering__

This algorithm produces a rating for a Business by a User by combining ratings of similar Users using the Pearson Correlation distance metric.   
RMSE Achieved: 1.09

__3. Item Based Collaborative Filtering__

This algorithm produces a rating for a Business by a User by combining ratings of similar Businesses for which User has given a rating.  
RMSE Achieved: 1.07

Note: Also used Default Voting scheme in the User Based and Item Based model to achieve fast and accurate predictions.
