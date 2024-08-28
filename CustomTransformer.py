# ==============================================================================
# Watermark:
# This code is contributed by Nabil
# For contributions or questions, please contact: shafanda.nabil.s@gmail.com
# ==============================================================================
#

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    A transformer that renames columns in a DataFrame based on a provided dictionary.
    
    Parameters:
        rename_dict (Dict[str, str]): Dictionary mapping old column names to new column names.
    """
    def __init__(self, rename_dict):
        self.rename_dict = rename_dict

    def fit(self, X, y=None):
        """
        Fit method (does nothing in this case).

        Args:
            X (pd.DataFrame): DataFrame to fit on.
            y (Optional[pd.Series]): Target values (not used).

        Returns:
            ColumnRenamer: Fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Renames columns in the DataFrame based on the provided dictionary.

        Args:
            X (pd.DataFrame): DataFrame with columns to rename.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        return X.rename(columns=self.rename_dict)

class ValueCreator(BaseEstimator, TransformerMixin):
    """
    A transformer that creates new features in a DataFrame based on provided operations.

    Attributes:
        operations (Optional[Dict[str, str]]): Dictionary mapping new column names to operations.
        feature_names_out (str): Determines how feature names are outputted.
    """
    def __init__(
            self, 
            operations=None, 
            feature_names_out="one-to-one",
    ):
        self.operations = operations
        self.feature_names_out = feature_names_out

    def fit(self, X, y=None):
        """
        Fit method (checks features).

        Args:
            X (pd.DataFrame): DataFrame to fit on.
            y (Optional[pd.Series]): Target values (not used).

        Returns:
            ValueCreator: Fitted transformer.
        """
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X):
        """
        Creates new features in the DataFrame based on provided operations.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with new features created.
        """
        for col_name, condition in self.operations.items():
            if isinstance(X, pd.DataFrame):
                X[col_name] = X.eval(condition)
            else:
                X = self._apply_condition_np(X, col_name, condition)
        self.feature_names_out_ = X.columns if isinstance(X, pd.DataFrame) else None
        return X
    
    def _apply_condition_np(self, X, col_name, condition):
        df = pd.DataFrame(X)
        df[col_name] = df.eval(condition)
        return df.values
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names of the transformed DataFrame.

        Args:
            input_features (Optional[List[str]]): Input feature names (not used).

        Returns:
            np.ndarray: Array of feature names.
        """
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, transform='pandas'):
        """
        Set output type for the transform method.

        Args:
            transform (str): Desired output type ('pandas').

        Returns:
            ValueCreator: Updated transformer.
        """
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)

class ValueConverter(TransformerMixin, BaseEstimator):
    """
    A transformer that converts values in a DataFrame or numpy array based on a mapping dictionary.

    Attributes:
        mapping_dict (Optional[Dict[Union[str, int], str]]): Dictionary mapping old values to new values.
        default_value (str): Default value for unmapped values.
        feature_names_out (str): Determines how feature names are outputted.
    """
    def __init__(
        self,
        mapping_dict=None,
        default_value="others",
        feature_names_out="one-to-one",
    ):
        self.mapping_dict = mapping_dict
        self.default_value = default_value
        self.feature_names_out = feature_names_out
   
    def fit(self, X, y=None):
        """
        Fit method (checks features).

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Data to fit on.
            y (Optional[pd.Series]): Target values (not used).

        Returns:
            ValueConverter: Fitted transformer.
        """
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        """
        Converts values in the DataFrame or numpy array based on the mapping dictionary.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Data to transform.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Transformed Data.
        """
        if self.mapping_dict is not None: 
            if isinstance(X, pd.DataFrame):
                return X.applymap(self._map_value)
            return np.vectorize(self._map_value)(X)
        return X
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names of the transformed DataFrame.

        Args:
            input_features (Optional[List[str]]): Input feature names (not used).

        Returns:
            np.ndarray: Array of feature names.
        """
        return np.asarray(self.feature_names_in_, dtype=object)
        
    def _map_value(self, value):
        return self.mapping_dict.get(value, self.default_value)

    def set_output(self, transform='pandas'):
        """
        Set output type for the transform method.

        Args:
            transform (str): Desired output type ('pandas').

        Returns:
            ValueConverter: Updated transformer.
        """
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)

class ValueClassifier(TransformerMixin, BaseEstimator):
    """
    A custom transformer for classifying values in a dataset based on conditions.

    Parameters:
        conditions (List[str], optional): List of conditions to apply for classification.
        choices (List[Union[str, int]], optional): List of choices corresponding to each condition.
        default (Union[str, int], optional): Default value to use when no conditions are met.
        feature_names_out (str, optional): Method to determine feature names for the output DataFrame.
        rename (str, optional): Name of the new feature column added to the DataFrame.

    Attributes:
        feature_names_out_ (Optional[Union[np.ndarray, None]]): Names of features after transformation.
    """
    def __init__(self, 
        conditions=None, 
        choices=None, 
        default="Unknown",
        feature_names_out="one-to-one",
        rename="status",
    ):
        self.conditions = conditions
        self.choices = choices
        self.default = default
        self.feature_names_out = feature_names_out
        self.rename = rename
   
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. This method is a no-op for this transformer.
        
        Parameters:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target values (ignored in this transformer).

        Returns:
            self: Returns the instance itself.
        """
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X):
        """
        Transform the data according to the specified conditions and choices.

        Parameters:
            X (pd.DataFrame): The input data to transform.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Transformed data.
        """
        if self.conditions is not None and self.choices is not None: 
            if isinstance(X, pd.DataFrame):
                return self._transform_df(X)
            return self._transform_array(X)
        return X
    
    def _transform_df(self, X):
        X = X.copy()
        condition = [self._apply_condition(cond, X) for cond in self.conditions]
        X[self.rename] = np.select(condition, self.choices, default=self.default)
        self.feature_names_out_ = X.columns if isinstance(X, pd.DataFrame) else None
        return X
    
    def _transform_array(self, X):
        if X.ndim == 1:
            return np.array([self.default for val in X])
        elif X.ndim == 2:
            condition = [self._apply_condition(cond, pd.DataFrame(X)) for cond in self.conditions]
            return np.select(condition, self.choices, default=self.default)
        else:
            raise ValueError("Unsupported array shape.")

    def _apply_condition(self, condition, X):
        if isinstance(X, pd.DataFrame):
            return X.eval(condition)
        else:
            raise ValueError("Unsupported input type for condition application.")
    
    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names after transformation.

        Parameters:
            input_features (Optional[List[str]]): The feature names of the input data (ignored).

        Returns:
            np.ndarray: Array of feature names.
        """
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, transform='pandas'):
        """
        Set the output format for the transformation.

        Parameters:
            transform (str): Output format, 'pandas' or others.

        Returns:
            self: Returns the instance itself.
        """
        if transform == 'pandas':
            self.transform = self._wrapped_transform_pandas
        return self
    
    def _wrapped_transform_pandas(self, X): 
        return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)