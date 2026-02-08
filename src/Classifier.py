import pickle
import os
import pandas as pd

class LiverDiseaseClassifier:
    """Base class for the Liver Disease Prediction pipeline."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_path = 'liver_model.pickle'
        self.df = None
        self.model = None
        # Placeholders for train/test data
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_saved_model(self):
        """Attempts to load a pre-trained model from the disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Success: Model loaded from {self.model_path}")
            return True
        print("Info: No saved model found on disk.")
        return False