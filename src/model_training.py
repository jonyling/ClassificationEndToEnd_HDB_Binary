# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTraining:
    """
    A class used to train and evaluate machine learning models on HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = {
            "target_column": config.get("target_column", "price_category"),
            "val_test_size": config.get("val_test_size", 0.2),
            "val_size": config.get("val_size", 0.5),
            "param_grid": config.get("param_grid", {
                "ridge_tuned": {"regressor__alpha": [0.1, 1.0, 10.0]}
            }),
            "cv": config.get("cv", 5),
            "scoring": config.get("scoring", "f1")
        }
        self.preprocessor = preprocessor
        if not isinstance(self.preprocessor, ColumnTransformer):
            raise ValueError("preprocessor must be a ColumnTransformer instance")
            logging.info(f"Initialized ModelTraining with scoring: {self.config['scoring']}")

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
            A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        try:
            X = df.drop(columns=self.config["target_column"])
            y = df[self.config["target_column"]]
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.config["val_test_size"], random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=self.config["val_size"], random_state=42
            )
            logging.info("Data split into train, validation, test sets.")
            return X_train, X_val, X_test, y_train, y_val, y_test
        except KeyError as e:
            logging.error(f"Error splitting data: {e}")
            raise

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "ridge": RidgeClassifier(random_state=42),
        }
        pipelines = {}
        metrics = {}

        try:
            for model_name, model in models.items():
                pipeline = Pipeline(
                    steps=[("preprocessor", self.preprocessor), ("classifier", model)]
                )
                pipeline.fit(X_train, y_train)
                pipelines[model_name] = pipeline
                metrics[model_name] = self._evaluate_model(
                    pipeline, X_val, y_val, model_name
                )
        except Exception as e:
            logging.error(f"Error training baseline models: {e}")
            raise

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for Ridge and Lasso models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        param_grid = self.config["param_grid"]
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        if scoring is None:
            logging.warning("Scoring is None, defaulting to 'f1'")
            scoring = "f1"

        models = {"ridge_tuned": RidgeClassifier(random_state=42)}

        try:
            for model_name, model in models.items():
                logging.info(f"Scoring: {scoring}, Param Grid: {param_grid.get(model_name, {})}")
                pipeline = Pipeline(
                    steps=[("preprocessor", self.preprocessor), ("classifier", model)]
                )
                grid_search = GridSearchCV(
                    pipeline, param_grid.get(model_name, {}), cv=cv, scoring=scoring, n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                tuned_models[model_name] = grid_search.best_estimator_
                tuned_metrics[model_name] = self._evaluate_model(
                    tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
                )
        except Exception as e:
            logging.error(f"Error tuning models: {e}")
            raise

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        try:
            y_test_pred = model.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_test_pred),
                "F1": f1_score(y_test, y_test_pred),
                "ROC-AUC": roc_auc_score(y_test, y_test_pred),
            }
            logging.info(f"Final Test Metrics for {model_name}:")
            for metric_name, metric_value in metrics.items():
                logging.info(f"{metric_name}: {metric_value}")
            return metrics
        except NotFittedError as e:
            logging.error(f"Model not fitted for evaluation: {e}")
            raise
        except Exception as e:
            logging.error(f"Error evaluating final model: {e}")
            raise

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        try:
            y_val_pred = model.predict(X_val)
            metrics = {
                "Accuracy": accuracy_score(y_val, y_val_pred),
                "F1": f1_score(y_val, y_val_pred),
                "ROC-AUC": roc_auc_score(y_val, y_val_pred),
            }
            logging.info(f"{model_name} Validation Metrics:")
            for metric_name, metric_value in metrics.items():
                logging.info(f"{metric_name}: {metric_value}")
            return metrics
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
            raise