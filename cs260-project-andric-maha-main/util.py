import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from track import Track

'''
Authors: Maha Attique, Andric Brena
Description: This class contains utility functions for use in running Gaussian Naive Bayes,
             Information Gain, and PCA.
Data: 12/13/23
'''
def normalize(dataset):
    '''
    Normalizes each column in the dataset.
    Params:
        dataset: the dataframe object to be normalized
    Return:
        dataset: with its values normalized for each column
    '''
    #for each feature, subtract the values by the sample mean and then divide by the std
    for feature in dataset:
        arr_vals = dataset[feature].to_numpy()
        arr_vals = (arr_vals - arr_vals.mean()) / (arr_vals.std())
        dataset[feature] = arr_vals
    return dataset

def calculate_average_info_gain(features, target):
    """
    Calculate average information gain for each continuous feature.

    Parameters:
    - features (pd.DataFrame): Features for training.
    - target (pd.Series): Target variable for training.

    Returns:
    - info_gain_dict: Dictionary containing average information gain for each continuous feature.
    """
    info_gain_dict = {} #declares the return variable
    continuous_features = features.select_dtypes(include=['float64']).columns #list of continuous features

    #goes through each continous feature to calculate its average information gain
    for feature in continuous_features:
        intervals = create_intervals(features[feature]) #list of intervals
        info_gain_list = [] #list containing information gain for each interval

        # Calculate entropy of the entire dataset (before the split)
        entropy_before_split = entropy(target.value_counts(normalize=True), base=2)

        #calculates the information gain for each interval
        for interval in intervals:
            # Create a binary variable indicating whether the value is in the interval
            binarized_feature = features[feature].apply(lambda x: 1 if interval[0] <= x <= interval[1] else 0)

            # Calculate entropy of the child node (after the split)
            entropy_after_split = entropy(target[binarized_feature == 1].value_counts(normalize=True), base=2)

            # Calculate information gain
            info_gain = entropy_before_split - entropy_after_split
            info_gain_list.append(max(info_gain, 0))  # Ensure non-negative values

        # Calculate average information gain for the continuous feature
        average_info_gain = np.mean(info_gain_list)
        info_gain_dict[feature] = average_info_gain

    return info_gain_dict

def create_intervals(series):
    """
    Create intervals for a continuous series.

    Parameters:
    - series (pd.Series): Continuous variable.

    Returns:
    - list: List of intervals.
    """
    min_val = series.min() #minimum value
    max_val = series.max() #maximum value
    num_intervals = 5  
    interval_width = (max_val - min_val) / num_intervals

    #creates the intervals
    intervals = [(min_val + i * interval_width, min_val + (i + 1) * interval_width) for i in range(num_intervals)]

    return intervals

def predict_class_for_feature(features_train, features_test, target_train, target_test, selected_class, feature_importance):
    '''
    Utilizes the weights calculated from GaussianNB to select the most important feature for a specified class.
    Uses this feature to train and evaluate a single-feature model based on binary classification where the specified 
    class has label 1 and all other classes have label 0.
    Parameters:
        - features_train (pd.DataFrame): Features for training.
        - features_test (pd.DataFrame): Features for testing.
        - target_train (pd.Series): Target variable for training.
        - target_test (pd.Series): Target variable for testing.
        - selected_class: String representing the class for binary classification
        - feature_importance: Matrix containing the feature weight for each class
    '''
    # Convert the NumPy array to a dictionary
    feature_importance_dict = dict(zip(features_train.columns, feature_importance[selected_class]))

    # Get the feature with the highest importance for the selected class
    selected_feature = max(feature_importance_dict, key=feature_importance_dict.get)
    print(f"Selected feature for predicting {selected_class}: {selected_feature}")

    # Create a binary target variable indicating whether the selected class is present or not
    binary_target_train = (target_train == selected_class).astype(int)
    binary_target_test = (target_test == selected_class).astype(int)

    # Train and evaluate the model using only the selected feature for binary classification
    print("Accuracy for predicting", selected_class, " using ", selected_feature, ":")
    train_and_evaluate(features_train[[selected_feature]], features_test[[selected_feature]], binary_target_train, binary_target_test)

def visualize_feature_importance(features_train, target_train, genres_dict, color_dict):
    """
    Visualize feature importance based on Gaussian Naive Bayes model for multiclass classification.

    Parameters:
    - features_train (pd.DataFrame): Features for training.
    - target_train (pd.Series): Target variable for training.
    - genres_dict (dict): Dictionary mapping class labels to genre names.

    Return:
    - feature_imp_dict: dictionary pairing each class with their list of weights
    """
    #Runs Gaussian Naive Bayes on the provided training dataset
    model = GaussianNB()
    model.fit(features_train, target_train)

    # Get feature importance scores for each class
    feature_importance = model.theta_  # Use mean values for each class

    # Plot absolute values of feature importance for each class
    feature_imp_dict = {}
    plt.figure(figsize=(12, 8))
    for class_label, class_importance in enumerate(feature_importance):
        genre_name = genres_dict[class_label]
        color = color_dict[genre_name]
        feature_imp_dict[genre_name] = np.abs(class_importance)
        plt.bar(features_train.columns, np.abs(class_importance), label=f'{genre_name}', color=color)
    plt.xlabel('Feature')
    plt.ylabel('Absolute Feature Importance')
    plt.title('Absolute Feature Importance based on Gaussian Naive Bayes (Multiclass)')
    plt.legend()
    plt.show()
    return feature_imp_dict

def perform_pca(features_train, features_test, target_train, target_test, color_dict, selected_classes=None, transparency=0.15):
    """
    Perform PCA and plot the results for specified classes.

    Parameters:
    - features_train (pd.DataFrame): Features for training.
    - features_test (pd.DataFrame): Features for testing.
    - target_train (pd.Series): Target variable for training.
    - target_test (pd.Series): Target variable for testing.
    - color_dict (dict): Dictionary mapping class labels to colors.
    - selected_classes (list or None): List of class labels to include in the plot. If None, include all classes.
    - transparency: float measuring how transparent each data point will be on the PCA plot

    Returns:
    - pd.DataFrame: DataFrame with principal components and class labels for visualization.
    """
    # Concatenate the training and testing features
    all_features = pd.concat([features_train, features_test], axis=0)

    # Standardize the features
    scaler = StandardScaler()
    all_features_standardized = scaler.fit_transform(all_features)

    # Create a PCA instance with the desired number of components
    n_components = 2  # Choose the number of principal components
    pca = PCA(n_components=n_components)

    # Fit the PCA model and transform the standardized features
    principal_components = pca.fit_transform(all_features_standardized)

    # Create a DataFrame with the principal components and target labels for visualization
    df_pca = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    df_pca['Target'] = pd.concat([target_train, target_test], axis=0).reset_index(drop=True)

    # Plot the PCA results for specified classes
    plt.figure(figsize=(11, 8))

    #determines whether to plot all classes or a specified list of classes
    if selected_classes is None:
        classes_to_plot = df_pca['Target'].unique()
    else:
        classes_to_plot = selected_classes

    #plots PCA results
    for class_label in classes_to_plot:
        indices_to_keep = df_pca['Target'] == class_label
        color = color_dict[class_label]
        plt.scatter(df_pca.loc[indices_to_keep, 'PC1'],
                    df_pca.loc[indices_to_keep, 'PC2'],
                    c=color,
                    label=str(class_label),
                    alpha=transparency)
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.title('PCA of Track Features')
    plt.legend()
    plt.show()

def create_color_dictionary(classes):
    '''
    Creates a dictionary with key:class and value:color where each value is a distinct color
    Parameter:
        classes: list of classes
    Return:
        color_dict: dictionary assigning a color to each class
    '''
    unique_classes = np.unique(classes) #stores the distinct classes
    num_classes = len(unique_classes) #number of distinct classes

    #creates dictionary assigning each class to a specific color
    color_dict = {class_label: plt.cm.rainbow(i / num_classes) for i, class_label in enumerate(unique_classes)}
    return color_dict

def get_tracks(df):
    """
    Create a dictionary of Track objects from a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing track information.

    Returns:
    - dict: Dictionary of Track objects with track_id as keys.
    """
    tracks_dict = {} #declares tracks dictionary

    #iterates through the dataframe to fill tracks_dict
    #with Track objects as values and their track id as their key
    for index, row in df.iterrows():
        track_id = row['track_id']
        features_dict = row.drop(['track_id', 'genre_top']).to_dict()
        target_value = row['genre_top']
        tracks_dict[track_id] = Track(features_dict, target_value)
    return tracks_dict

def prepare_data(df):
    """
    Prepare features and target variables from a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing track information.

    Returns:
    - tuple: Tuple of features and target variables.
    """
    features = normalize(df.drop(['track_id', 'genre_top', 'title'], axis=1)) #normalizes feature dataframe
    target = df['genre_top'] #grabs the label/class for each example
    return features, target

def train_and_evaluate(features_train, features_test, target_train, target_test):
    """
    Train and evaluate a Gaussian Naive Bayes model.

    Parameters:
    - features_train (pd.DataFrame): Features for training.
    - features_test (pd.DataFrame): Features for testing.
    - target_train (pd.Series): Target variable for training.
    - target_test (pd.Series): Target variable for testing.
    """
    #Runs Gaussian Naive Bayes on the training dataset
    model = GaussianNB()
    model.fit(features_train, target_train)
    
    predictions = model.predict(features_test) #list of predictions
    # print('pred:', predictions.size)
    accuracy = accuracy_score(target_test, predictions) #accuracy score using the predictions and the true values
    cm = confusion_matrix(target_test, predictions) #builds the confusion matrix using the GaussianNB model
  
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=np.unique(target_train), yticklabels=np.unique(target_train), cmap='coolwarm')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()
    print(f"Accuracy: {accuracy}\n")

def get_tracks_data(feature_df, tracks_df):
    """
    Get tracks data, filter tracks, and merge with features.

    Parameters:
    - feature_df (pd.DataFrame): DataFrame containing acoustic features.
    - tracks_df (pd.DataFrame): DataFrame containing track information.

    Returns:
    - tuple: Tuple of DataFrames for training and testing.
    """
    # List of tracks that also have information regarding their acoustic features
    list_tracks = feature_df['track_id'].to_numpy()

    # Reducing the set of tracks to those that have information on their acoustic features
    condensed_tracks_df = tracks_df.loc[tracks_df['track_id'].isin(list_tracks)]

    # Extracting only required columns from the tracks DataFrame
    selected_columns = ['track_id', 'genre_top', 'title']
    extracted_tracks_df = condensed_tracks_df[selected_columns]

    # Merging the track DataFrame with its corresponding features DataFrame
    merged_df = pd.merge(extracted_tracks_df, feature_df, on='track_id', how='inner')

    # Drop rows with missing values
    merged_df.dropna(inplace=True)

    # Splitting the merged tracks into train and test datasets
    tracks_train_df = merged_df[:7000]
    tracks_test_df = merged_df[7000:]

    return tracks_train_df, tracks_test_df

def get_top_genres(genres_df):
    """
    Get a dictionary of top genres.

    Parameters:
    - genres_df (pd.DataFrame): DataFrame containing genre information.

    Returns:
    - dict: Dictionary of top genres with genre_id as keys.
    """
    parent = genres_df['parent'] #dataframe containing the parent genres for each genre/subgenre
    name = genres_df['title'] #string representation of each genre
    genre_id = genres_df['genre_id'] #int id for each genre
    top_genres = {} #declares the dictionary to contain the genres at the top level
    
    #iterates through the parent dataframe, filling top_genres with those
    #whose parent genre == 0 i.e. genres at the top level
    i = 0
    for i in range(len(parent)):
        if parent[i] == 0:
            top_genres[genre_id[i]] = name[i]
        i += 1
    return top_genres

def createRadarElement(row, feature_cols):
    '''
    Creates a radar plot for a single row of data.
    Parameters:
        row: a list containing the feature values for a track
        feature_cols: list containing the features
    Return:
        Scatterpolar object
    '''
    #Creates a Scatterpolar object
    return go.Scatterpolar(
        r = row[feature_cols].values.tolist(), #grabs the feature values
        theta = feature_cols, 
        mode = 'lines', 
        name = row['title']) #labels the plot using the track name


def get_radar_plot(tracks_df, genre, num_tracks):
    '''
    Creates a radar plot using a specified amount of tracks from a given genre.
    Parameters:
        tracks_df: dataframe containing tracks
        genre: the genre to select tracks from
        num_tracks: the number of tracks to plot 
    '''
    genre_tracks = tracks_df.loc[tracks_df['genre_top'] == genre] #dataframe containing tracks with the specified genre
    features = tracks_df.columns.drop(['track_id', 'genre_top', 'title']).to_numpy() #feature value matrix
    genre_tracks.update(normalize(genre_tracks[features])) #updates genre_tracks with the normalized feature values

    #Creates and shows the radar plot 
    current_data = list(genre_tracks.sample(num_tracks).apply(createRadarElement, axis=1, args=(features, )))  
    fig = go.Figure(current_data, )
    fig.show()  
   
