'''
Authors: Maha Attique
Description: This file defines a Track object containing the auditory features and genre for a track.
Data: 12/13/23
'''
class Track:
    """
    Class to represent a track with its features and target genre.
    """
    def __init__(self, features, target):
        """
        Initializes a Track object.

        Parameters:
        - features (dict): Dictionary containing track features.
        - target (str): Target genre.
        """
        self.x = features
        self.y = target

    def __str__(self):
        """
        Custom string representation for the Track object.

        Returns:
        - str: String representation of the Track.
        """
        return f"Track Features: {self.x}, Target Genre: {self.y}"