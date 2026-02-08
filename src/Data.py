import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .Classifier import LiverDiseaseClassifier


class DataAnalysis(LiverDiseaseClassifier):

    def load_and_preprocess(self):
        """Load dataset and perform initial encoding and cleaning."""
        self.df = pd.read_csv(self.file_path)

        # Encoding gender to numeric
        self.df['gender'] = self.df['gender'].map({'Male': 1, 'Female': 0})

        # Remapping target: 1/Sick = 1, 2/Healthy = 0
        self.df['is_patient'] = self.df['is_patient'].map({1: 1, 2: 0})

        # Handling missing values
        self.df.dropna(inplace=True)

    def exploratory_analysis(self):
        """Generate correlation matrix heatmap for data insights."""
        plt.figure(figsize=(10, 8))
        correlation = self.df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()

    def boxplot(self):
        plot_df = self.df[['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins',
                           'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']].copy()

        cols_to_log = ['tot_bilirubin', 'direct_bilirubin', 'sgpt', 'sgot', 'alkphos']

        for col in cols_to_log:  # logarithmization
            plot_df[col] = np.log1p(plot_df[col])

        plot_df.plot(
            kind='box',
            subplots=True,
            layout=(3, 3),
            figsize=(15, 12),
            sharey=False,
        )

        plt.tight_layout()
        plt.show()