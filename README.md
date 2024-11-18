## CS260 Project: Lab Notebook
Maha and Andric: 11-28-23 (1hr)
    We made sure we can both access the repository and push any changes. 
    Given that we are working with large datasets that caused errors when commiting them, a Google Drive folder was made instead where we can both view them. 

Maha and Andric: 12-3-23 (30min)
    We met to discuss the data we wish to focus on.
    We determined that genres.csv, echonest.csv, and tracks.csv will best help us in realizing our project goals. Specifically, echonest.csv provides us with each track's auditory features, tracks.csv tells us the each track's genre ID at the top level, and genres.csv acts as a reference file that specifies the title of a genre given its ID. Our next goal is to determine how to go about grabbing subsets of the relevant data in order to work with them in VSC.

Maha and Andric: 12-4-23 (1hr)
    Made changes to the headings of each file to simplify the amount of headers. Decided to use pandas to represent each csv file as a dataframe. Reduced the amount of tracks to only consist of the ones that have auditory features.

Maha and Andric: 12-5-23 (30min) 
    Simplified our tracks dataset to only contain tracks that have one single parent genre. This left us with 9355 examples, and from there, we split the examples into training and testing datasets. Considering our desire to apply Naive Bayes and the fact that our feature values are continuous, we're planning on utilizing the available package for Gaussian Naive Bayes algorithm.

Maha: 12-6-23 (3 hours)
    Cleaned up the code and added methods. Formatted the data to use with the sk-learn Gaussian Bayes model. Ran Gaussian bayes without feature selection.

Maha: 12-7-23 (4 hours)
    Tried implementing Gaussian Bayes with feature selection based on maximum classification accuracy and mutual information classification (substituting information gain) to compare model accuracies at predicting genres. Implemented PCA. 

Andric: 12-10-2023 (3 hours)
    Worked on data normalization and radar plots. Changed color transparency for PCA. Added a confusion matrix represented by a heat map for the Gaussian Naive Bayes model.

Maha: 12-12-2023 (1 hour)
    Worked on representing all classes in PCA and the feature importance plots. Met with the professor to disucss the project and presentation.

Maha and Andric: 12-12-2023 (2 hours)
    Debugging and fixing plots to make sure they are well represented. Added a color dictionary to ensure uniformity across the visualizations. Fized the radar plots to accurately represent the genres.

Maha: 12-13-2023 (2.5 hours)
    Changed continous features into discrete and implemented information gain to see which feature predicts genre the best across all classes. 

Andric: 12-13-2023 (4 hours)
    Streamlined the integration of radar plots. Reorganized the code into separate files and added comments for each one.

Andric: 12-14-2023 (2 hours)
    Created a folder for our figures and included references containing the dataset and papers that informed my initial exploration of the research area. Made minor changes to the PCA to better reflect our findings by highlighting the genres that were least represented and those that were most represented in the dataset. Added print statements to make it easier to follow along as the program gets executed. Put all relevant plots/matrices in the figures folder. Also included the pdf of our slides presentation.

Maha: 12/15/2023 (1 hour):
    Added some more information in Main.py. Added instructions to reproduce our results. Added the link to the Driive folder containing our cleaned data files. 


**Instructions on how to run**
1) Download the cleaned data files from the following link: https://drive.google.com/drive/folders/1CJD1V90z4Ax_YIQgsJpNhu-LT0xewIPY?usp=sharing 

2) In Main.py, for the following dataframes, change the path of the data files to where you have saved them:
    feature_df = pd.read_csv('path_to_Cleaned_Features.csv')
    tracks_df = pd.read_csv('path_to_Cleaned_Tracks.csv')
    genres_df = pd.read_csv('path_to_genres.csv')

3) Make sure you have all the libraries that we are using downloaded. They can be viewed in util.py, and are also listed below: 
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

4) If all looks well, and you have all the files correctly stored together, go ahead and use terminal in vscode to run main.py:
    python3 main.py
That should be all that you need to do to reproduce our results.


**References**
Defferrard, Benzi, Vandergheynst, et al. "FMA: A Dataset for Music Analysis." 18th International Society for Music Information Retrieval Conference (2017).

Defferrard, Mohanty, Carroll, et al. "Learning to recognize musical genre from audio." The 2018 Web Conference Companion (2018).

Github - fma repository
https://github.com/mdeff/fma#fma-a-dataset-for-music-analysis