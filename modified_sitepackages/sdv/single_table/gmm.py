"""Wrapper around TVAE model."""
import numpy as np
import pandas as pd
import plotly.express as px
from sdmetrics import visualization

from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns

from sklearn.mixture import GaussianMixture

from tqdm import tqdm
import wandb

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if pd.api.types.is_categorical_dtype(dtype)
    ]
    if category_cols:
        categoricals = "', '".join(category_cols)
        error_msg = (
            f"Columns ['{categoricals}'] are stored as a 'category' type, which is not "
            "supported. Please cast these columns to an 'object' to continue."
        )
        raise InvalidDataTypeError(error_msg)



class GMM(BaseSynthesizer):
    """Gaussian Mixture Model."""

    def __init__(
        self,
        covariance_type:str = "full",
        max_iter: int = 100,
        n_components: int = 15,
        init_params: str = 'kmeans',
        verbose=False,
        use_wandb= False
    ):

        self._covariance_type = covariance_type
        self._max_iter = max_iter
        self._n_components = n_components
        self._init_params = init_params

        self.verbose = verbose
        self._use_wandb = use_wandb

        self._isBuilt = False

    def _built(self):
        self.model = GaussianMixture(n_components=self._n_components, covariance_type=self._covariance_type, max_iter= self._max_iter,
                                     init_params=self._init_params, random_state= 42, verbose= self.verbose)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the GMM Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        if not self._isBuilt:
            self._built()
            self._isBuilt = True

        self.model.fit(train_data)

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        fake = self.model.sample(n_samples= samples)[0]

        return self.transformer.inverse_transform(fake)

class GMMSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``TVAE`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        compress_dims (tuple or list of ints):
            Size of each hidden layer in the encoder. Defaults to (128, 128).
        decompress_dims (tuple or list of ints):
           Size of each hidden layer in the decoder. Defaults to (128, 128).
        l2scale (int):
            Regularization term. Defaults to 1e-5.
        batch_size (int):
            Number of data samples to process in each step.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        loss_factor (int):
            Multiplier for the reconstruction error. Defaults to 2.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }

    def __init__(self, metadata, covariance_type:str = "full", max_iter: int = 100, n_components: int = 15,
        init_params: str = 'kmeans', enforce_min_max_values=True, enforce_rounding=True, verbose=False, use_wandb= False):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_components = n_components
        self.init_params = init_params

        self._model_kwargs = {
            'covariance_type': covariance_type,
            'max_iter': max_iter,
            'n_components': n_components,
            'init_params': init_params,
            'verbose': verbose,
            'use_wandb': use_wandb
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            processed_data,
            transformers
        )
        self._model = GMM(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("GMMSynthesizer doesn't support conditional sampling.")
