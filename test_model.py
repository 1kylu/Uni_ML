import unittest
import numpy as np
import os
import pickle


class TestLiverModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ten kod wykona się raz przed wszystkimi testami - ładujemy model"""
        cls.model_path = 'liver_model.pickle'
        if os.path.exists(cls.model_path):
            with open(cls.model_path, 'rb') as f:
                cls.model = pickle.load(f)
        else:
            cls.model = None

    def test_model_exists(self):
        """Sprawdza, czy plik modelu w ogóle istnieje"""
        self.assertTrue(os.path.exists(self.model_path), "Plik modelu .pickle nie został znaleziony!")

    def test_prediction_output_shape(self):
        """Sprawdza, czy model zwraca poprawny format danych (0 lub 1)"""
        if self.model:
            # Tworzymy przykładowe dane dla jednego pacjenta (10 cech)
            dummy_patient = np.array([[45, 1, 1.2, 0.5, 200, 45, 50, 6.8, 3.3, 0.9]])
            prediction = self.model.predict(dummy_patient)

            # Sprawdzamy, czy wynik to tablica z jednym elementem
            self.assertEqual(len(prediction), 1)
            # Sprawdzamy, czy wynik to 0 lub 1
            self.assertIn(prediction[0], [0, 1])

    def test_probability_range(self):
        """Sprawdza, czy prawdopodobieństwo mieści się w zakresie 0-1"""
        if self.model:
            dummy_patient = np.array([[30, 0, 0.8, 0.2, 150, 20, 25, 7.0, 4.0, 1.2]])
            probs = self.model.predict_proba(dummy_patient)

            # Sprawdzamy, czy prawdopodobieństwa sumują się do 1 (lub blisko 1)
            self.assertAlmostEqual(np.sum(probs[0]), 1.0, places=5)
            # Sprawdzamy, czy wartości są nieujemne
            self.assertTrue(np.all(probs >= 0))
