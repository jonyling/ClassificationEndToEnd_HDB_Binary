# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        
        self.config = {
            "numerical_features": config.get("numerical_features", ['floor_area_sqm', 'remaining_lease_months', 'lease_commence_date', 'year']),
            "nominal_features": config.get("nominal_features", ['month', 'town_name', 'flatm_name', 'storey_range']),
            "ordinal_features": config.get("ordinal_features", ['flat_type']),
            "passthrough_features": config.get("passthrough_features", []),
            "flat_type_categories": config.get("flat_type_categories", ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'MULTI-GENERATION', 'EXECUTIVE'])  # Added default categories
        }
        # Defer preprocessor creation
        self.preprocessor = None # Initialize as None, create on demand

        self.preprocessor = self._create_preprocessor()
        logging.info("Preprocessor initialized successfully.")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the new data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        try:
            # Validate required columns
            required_columns = ['flat_type', 'lease_commence_date', 'remaining_lease', 'town_id', 'flatm_id', 'town_name', 'flatm_name', 'month', 'storey_range', 'floor_area_sqm', 'id']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Absolute negative years values
            df['lease_commence_date'] = df['lease_commence_date'].abs()

            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Standardize flat_type values and validate against categories
            df["flat_type"] = df["flat_type"].replace({
                "FOUR ROOM": "4 ROOM"  # Standardize to match category list
            })

            # Validate that all flat_types are in the configured categories
            invalid_categories = set(df["flat_type"].dropna().unique()) - set(self.config["flat_type_categories"])
            if invalid_categories:
                logging.warning(f"Invalid flat_type categories found: {invalid_categories}. Assigning to 'Unknown'.")
                df.loc[df["flat_type"].isin(invalid_categories), "flat_type"] = "Unknown"

            # Handle lease_commence_date anomalies
            negative_years = df[df["lease_commence_date"] < 0]
            if not negative_years.empty:
                logging.warning(f"Found negative lease_commence_date values: {negative_years['lease_commence_date'].unique()}. Correcting to positive.")
            df["lease_commence_date"] = df["lease_commence_date"].abs()
            df = df[df["lease_commence_date"].between(1960, 2025)]  # Validate year range

            # Convert storey_range to average numerical value
            df["storey_range"] = df["storey_range"].apply(self._convert_storey_range)

            # Fill missing names using ID mappings
            df = self._fill_missing_names(df, "town_id", "town_name")
            df = self._fill_missing_names(df, "flatm_id", "flatm_name")

            # Convert month to year and month columns
            df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m", errors='coerce')
            df["year"] = df["year_month"].dt.year
            df["month"] = df["year_month"].dt.month
            df.drop(columns=["year_month"], inplace=True)

            # Extract remaining lease months
            df["remaining_lease_months"] = df["remaining_lease"].apply(self._extract_lease_info)
            df["remaining_lease_months"] = df["remaining_lease_months"].fillna(df["remaining_lease_months"].median())

            # Drop unnecessary columns
            df.drop(columns=["id", "town_id", "flatm_id", "block", "street_name", "remaining_lease"], inplace=True)

            # Impute missing numerical values
            df[self.config["numerical_features"]] = df[self.config["numerical_features"]].fillna(df[self.config["numerical_features"]].median())

            # Encode price_category to numeric values
            label_mapping = {'Below Median': 0, 'Above Median': 1}
            df["price_category"] = df["price_category"].map(label_mapping)
            self.config["target_column"] = "price_category"  # Ensure target column is updated

            logging.info("Data cleaning completed.")
            return df
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise
   
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        def get_preprocessor(self) -> ColumnTransformer:
            if self.preprocessor is None:
                try: 
                    self.preprocessor = self._create_preprocessor()
                except Exception as e:
                    logging.error(f"Failed to create preprocessor: {e}")
                    raise
            return self.preprocessor
        
        
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("oneshot", OneHotEncoder(handle_unknown="ignore"))]
        )
        # Define the ordinal categories for flat_type
        flat_type_categories = self.config["flat_type_categories"]

        # Define ordinal features to be ordinally encoded.
        ordinal_features = self.config["ordinal_features"]

        # Create an ordinal transformer pipeline
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(categories=[flat_type_categories],
                                    handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="drop",  # Drop any remaining columns not specified
            n_jobs=-1,
        )
        return preprocessor

    @staticmethod
    def _convert_storey_range(storey_range: str) -> float:
        """
        Converts a storey range string into its average numerical value.

        Args:
        -----
        storey_range (str): A string representing a range of storeys, in the format 'XX TO YY'.

        Returns:
        --------
        float: The average value of the two storeys in the range.
        """
        try:
            range_values = storey_range.split(" TO ")
            return (int(range_values[0]) + int(range_values[1])) / 2
        except (ValueError, IndexError) as e:
            logging.warning(f"Invalid storey_range format: {storey_range}, returning 0")
            return 0.0

    @staticmethod
    def _fill_missing_names(
        df: pd.DataFrame, id_column: str, name_column: str
    ) -> pd.DataFrame:
        """
        Fills missing values in the 'name_column' using the 'id_column'.

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the columns to be filled.
        id_column (str): The name of the column containing the IDs.
        name_column (str): The name of the column containing the names to be filled.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values in 'name_column' filled.
        """
        missing_names = df[name_column].isna() | (df[name_column] == '')
        name_mapping = (
            df[[id_column, name_column]]
            .dropna(subset=[name_column])
            .drop_duplicates()
            .set_index(id_column)[name_column]
            .to_dict()
        )
        unmapped = df.loc[missing_names, id_column][~df.loc[missing_names, id_column].isin(name_mapping.keys())]
        if not unmapped.empty:
            logging.warning(f"Unmapped {id_column} values: {unmapped.unique()}")
        df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(name_mapping).fillna('Unknown')
        return df

    @staticmethod
    def _extract_lease_info(lease_str: str) -> int:
        """
        Converts lease information from a string format to total months.

        Args:
        -----
        lease_str (str): The remaining lease period as a string.

        Returns:
        --------
        int: The total number of months, or None if the input is NaN.
        """
        if pd.isna(lease_str):
            return None
        lease_str = str(lease_str).strip()
        years_match = re.search(r"(\d+)\s*years?", lease_str)
        months_match = re.search(r"(\d+)\s*months?", lease_str)
        if years_match:
            years = int(years_match.group(1))
            months = int(months_match.group(1)) if months_match else 0
        else:
            try:
                years = int(lease_str) if lease_str.isdigit() else 0
                months = 0
            except ValueError:
                logging.warning(f"Invalid lease format: {lease_str}, returning 0")
                return 0
        total_months = years * 12 + months
        return total_months