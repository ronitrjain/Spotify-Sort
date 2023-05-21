import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances_argmin_min

# Spotify API credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'

# Get access token
def get_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(auth_url, data=auth_data)
    access_token = response.json()['access_token']
    return access_token

# Search for songs
def search_songs(query, limit=50):
    search_url = 'https://api.spotify.com/v1/search'
    access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'q': query,
        'type': 'track',
        'limit': limit
    }
    response = requests.get(search_url, headers=headers, params=params)
    songs = response.json()['tracks']['items']
    return songs

# Retrieve audio features for a song
def get_audio_features(song_id):
    features_url = f'https://api.spotify.com/v1/audio-features/{song_id}'
    access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(features_url, headers=headers)
    audio_features = response.json()
    return audio_features

# Collect song details and audio features
def collect_songs_data(query, num_songs):
    songs_data = []
    while num_songs > 0:
        limit = min(num_songs, 50)  # API limit is 50 songs per request
        songs = search_songs(query, limit=limit)
        for song in songs:
            song_id = song['id']
            audio_features = get_audio_features(song_id)
            songs_data.append({
                'song_id': song_id,
                'name': song['name'],
                'artists': ', '.join([artist['name'] for artist in song['artists']]),
                'album': song['album']['name'],
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'valence': audio_features['valence'],
                # Add more features as per your requirements
            })
            num_songs -= 1
            if num_songs <= 0:
                break
    songs_df = pd.DataFrame(songs_data)
    return songs_data

# Remove outliers using z-score
def remove_outliers(dataframe, features):
    z_scores = np.abs((dataframe[features] - dataframe[features].mean()) / dataframe[features].std())
    filtered_df = dataframe[(z_scores < 3).all(axis=1)]  # Adjust the z-score threshold as needed
    return filtered_df

# Preprocess the data to normalize selected features
def preprocess_data(dataframe, features):
    # Standardize the selected features using StandardScaler
    scaler = StandardScaler()
    dataframe[features] = scaler.fit_transform(dataframe[features])
    return dataframe

#Custom Transformer for Feature Selection
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.feature_names]

# Apply K-means clustering to the data
def apply_kmeans(dataframe, features, num_clusters):
    X = dataframe[features].values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

# Determine optimal features using GridSearchCV
def determine_features(dataframe, num_clusters):
    pipeline = Pipeline([
        ('selector', FeatureSelector(feature_names=dataframe.columns[4:])),
        ('kmeans', KMeans(n_clusters=num_clusters, random_state=42))
    ])
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'selector__feature_names': [
            ['danceability', 'energy', 'valence'],
            ['danceability', 'energy', 'valence', 'acousticness'],
            ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness'],
        ]
    }
    
    # Perform GridSearchCV to determine the optimal features
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(dataframe, dataframe['cluster'])
    
    # Get the best feature combination
    best_features = grid_search.best_params_['selector__feature_names']
    
    return best_features

# Function to generate playlist based on K-means clustering
def generate_playlist(song, dataframe, features, num_clusters, num_songs=100):
    input_song = preprocess_data(pd.DataFrame([song]), features)
    cluster_labels = apply_kmeans(dataframe, features, num_clusters)
    input_cluster = pairwise_distances_argmin_min(input_song[features].values, dataframe[features].values)[0][0]
    cluster_songs = dataframe[dataframe['cluster'] == input_cluster].copy()
    cluster_songs = cluster_songs.sample(frac=1, random_state=42)  # Shuffle the songs
    cluster_songs = cluster_songs[cluster_songs['song_id'] != song['song_id']]
    playlist = cluster_songs.head(num_songs).copy()
    
    return playlist

# Example usage
query = 'pop'
num_songs = 10000
num_clusters = 5  # Specify the desired number of clusters

# Collect song details and audio features
songs_data = collect_songs_data(query, num_songs)

# Create a DataFrame from the collected data
songs_df = pd.DataFrame(songs_data)

# Remove outliers using z-score for selected features
features_to_remove_outliers = ['danceability', 'energy', 'valence']  # Add more features as needed
filtered_df = remove_outliers(songs_df, features_to_remove_outliers)

# Preprocess the data to normalize selected features
features_to_normalize = ['danceability', 'energy', 'valence']  # Add more features as needed
preprocessed_df = preprocess_data(filtered_df, features_to_normalize)

# Determine optimal features using GridSearchCV
best_features = determine_features(preprocessed_df, num_clusters)

# Apply K-means clustering to the preprocessed data using the optimal features
cluster_labels = apply_kmeans(preprocessed_df, best_features, num_clusters)

# Add cluster labels to the DataFrame
preprocessed_df['cluster'] = cluster_labels

# Generate a playlist based on a sample song
sample_song = preprocessed_df.sample(n=1).iloc[0]
playlist = generate_playlist(sample_song, preprocessed_df, best_features, num_clusters, num_songs=100)

# Print the resulting playlist
print(playlist[['name', 'artists']])