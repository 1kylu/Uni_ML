import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .Classifier import LiverDiseaseClassifier


class Result(LiverDiseaseClassifier):

    def evaluate_model(self):
        """Analyze model performance using key metrics and Confusion Matrix."""
        predictions = self.model.predict(self.X_test)

        # Results Visualization
        cm = confusion_matrix(self.y_test, predictions)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Healthy (0)', 'Sick (1)'],
                    yticklabels=['Healthy (0)', 'Sick (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix: Random Forest')
        plt.show()

        # Calculating Metrics
        acc = accuracy_score(self.y_test, predictions)
        rec = recall_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        prec = precision_score(self.y_test, predictions)

        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
