import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .Classifier import LiverDiseaseClassifier


class Train(LiverDiseaseClassifier):

    def prepare_datasets(self, test_size=0.2):
        """Split data into training and testing sets."""
        X = self.df.drop('is_patient', axis=1).to_numpy()
        y = self.df['is_patient'].to_numpy()

        # Stratify ensures the same class distribution in train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=3000, stratify=y)

    def train_random_forest(self, n_estimators=300, force_train=False):
        """Train the model if not already loaded or if forced."""
        if self.model is not None and not force_train:
            print("Skipping training: Model is already loaded.")
            return

        print("Training Random Forest...")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=2,
            class_weight='balanced',  # Handles class imbalance
            random_state=1450,
            n_jobs=-1  # Uses all available CPU cores
        )
        self.model.fit(self.X_train, self.y_train)
        # Saving to disk
        with open('liver_model.pickle', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model trained and saved to 'liver_model.pickle'")