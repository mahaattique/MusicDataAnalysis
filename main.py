from util import *
'''
Authors: Maha Attique, Andric Brena
Description: This file runs the entire data processing and model training pipeline.
Data: 12/13/23
'''

def main(): 
    # Creates a DataFrame of the acoustic features for each track if present
    feature_df = pd.read_csv('/Users/mahaattique/Desktop/Data Science/final_project/data/Cleaned_Features.csv')
    # feature_df = pd.read_csv('/Users/andri/github-classroom/haverford-cs/final_project/data/Cleaned_Features.csv')
    
    # DataFrame containing every track and their respective genres in the dataset
    tracks_df = pd.read_csv('/Users/mahaattique/Desktop/Data Science/final_project/data/Cleaned_Tracks.csv')
    # tracks_df = pd.read_csv('/Users/andri/github-classroom/haverford-cs/final_project/data/Cleaned_Tracks.csv')
    
    # DataFrame connecting each genre id with their string representation
    genres_df = pd.read_csv('/Users/mahaattique/Desktop/Data Science/final_project/data/genres.csv')
    # genres_df = pd.read_csv('/Users/andri/github-classroom/haverford-cs/final_project/data/genres.csv')

    # Retrieve tracks data and split into training and testing sets
    read_data = get_tracks_data(feature_df, tracks_df)
    tracks_train_df = read_data[0]
    tracks_test_df = read_data[1]


    #creates a radar plot of 25 random International songs
    #for now only one plot can be made at a time during program execution
    get_radar_plot(tracks_train_df, 'International', 25)

    # Retrieve top genres dictionary
    top_genres = get_top_genres(genres_df)

    # Create dictionaries of Track objects for training and testing sets
    tracks_train_dict = get_tracks(tracks_train_df)
    tracks_test_dict = get_tracks(tracks_test_df)

    # Prepare data for training and testing
    features_train, target_train = prepare_data(tracks_train_df)
    features_test, target_test = prepare_data(tracks_test_df)

    # Dictionary connecting each genre to a specific color for graphing purposes
    genre_dict = np.unique(target_train)
    color_dict = create_color_dictionary(genre_dict)

   # Train and evaluate the model
    print('First model utilizing every feature for multi-class classification')
    train_and_evaluate(features_train, features_test, target_train, target_test)

    # Perform PCA and plot the results
    print('Performing PCA for all classes')
    perform_pca(features_train, features_test, target_train, target_test, color_dict, None, 0.35)

    genre_dict = np.unique(target_train)
    color_dict = create_color_dictionary(genre_dict)

    # Perform PCA for first 8 classes and plot the results
    print('Performing PCA for the first 8 least dense classes')
    classes_1 = ['Pop','Instrumental', 'Jazz', 'Blues', 'Classical', 'Old-Time / Historic', 'Experimental', 'International']
    perform_pca(features_train, features_test, target_train, target_test, color_dict, classes_1, 0.35)

    # Perform PCA for Rock and Hip-hop and plot the results
    print('Performing PCA for rock and hip-hop')
    classes_2 = ['Rock', 'Hip-Hop' ]
    perform_pca(features_train, features_test, target_train, target_test, color_dict, classes_2, 0.35)

    # Perform PCA for rock, hip-hop, electronic, and experimental and plot the results
    print('Performing PCA for rock, hip-hop, electronic, and folk\n')
    classes_3 = ['Rock', 'Hip-Hop','Electronic', 'Folk' ]
    perform_pca(features_train, features_test, target_train, target_test, color_dict, classes_3, 0.35)

    # Print and visualize feature importance
    feature_importance = visualize_feature_importance(features_train, target_train, genre_dict, color_dict)

    # specific classes for prediction
    selected_class_1 = 'Old-Time / Historic'  
    selected_class_2 = 'Rock'  
    selected_class_3 = 'Hip-Hop'

    # Predict the classes for the selected features
    print("Second model based on a single feature for binary classification")
    predict_class_for_feature(features_train, features_test, target_train, target_test, selected_class_1, feature_importance)
    predict_class_for_feature(features_train, features_test, target_train, target_test, selected_class_2, feature_importance)
    predict_class_for_feature(features_train, features_test, target_train, target_test, selected_class_3, feature_importance)

    
    # Calculate average information gain for each continuous feature
    average_info_gain = calculate_average_info_gain(features_train, target_train)

        
    #Get features with highest and lowest information gain
    highest_info_gain_feature = max(average_info_gain, key=average_info_gain.get)
    lowest_info_gain_feature = min(average_info_gain, key=average_info_gain.get)

    print(f"Feature with Highest Information Gain: {highest_info_gain_feature}")
    print(f"Feature with Lowest Information Gain: {lowest_info_gain_feature}")

    # Train a Naive Bayes model using the feature with the highest information gain
    print('Third model based on the feature with highest information gain for multi-class classification')
    train_and_evaluate(features_train[[highest_info_gain_feature]], features_test[[highest_info_gain_feature]], target_train, target_test)

    # Train a Naive Bayes model using the feature with the lowest information gain\
    print('Fourth model based on the feature with lowest information gain for multi-class classification')
    train_and_evaluate(features_train[[lowest_info_gain_feature]],features_test[[lowest_info_gain_feature]], target_train, target_test)

    print('Fifth model based on danceability for multi-class classification')
    train_and_evaluate(features_train[['danceability']],features_test[['danceability']], target_train, target_test)

if __name__ == "__main__":
    main()
