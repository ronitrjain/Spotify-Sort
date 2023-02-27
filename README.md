Spotify Song Classification
This Python script collects song data from the Spotify API for a given set of genres and artists, extracts relevant features such as popularity, tempo, acousticness, danceability, and energy, and uses a decision tree classifier to predict the artist of a new song based on its features.

Getting Started
To use this script, you will need to obtain a client ID and client secret from the Spotify developer dashboard. Once you have the client ID and client secret, replace the placeholders in the script with your own credentials.

You will also need to install the following Python packages:

spotipy
pandas
scikit-learn
You can install these packages using pip:

Copy code
pip install spotipy pandas scikit-learn
Usage
To run the script, simply execute it using Python:

Copy code
python spotify_classification.py
The script will collect song data for the specified genres and artists, extract relevant features, normalize the data, and train a decision tree classifier on the training set. It will then evaluate the performance of the model on the testing set and print a classification report. Finally, it will use the model to predict the artist of a new song based on its features.

Customization
You can customize the genres and artists of interest by modifying the genres and artists variables in the script. You can also adjust the number of songs collected per artist and genre combination by modifying the limit parameter in the sp.search method call.

You can modify the features used for classification by adjusting the list of feature names in the X_train and X_test variables. You can also adjust the normalization method by modifying the scaler variable.

You can customize the machine learning algorithm used for classification by replacing the DecisionTreeClassifier class with another classification algorithm from scikit-learn.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
