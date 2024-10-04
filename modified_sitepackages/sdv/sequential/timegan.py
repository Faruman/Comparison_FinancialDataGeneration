"""DOPPELGANGER Synthesizer class."""

import inspect
import logging
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

from .._utils import _cast_to_iterable, _groupby_list
from ..errors import SamplingError, SynthesizerInputError
from ..metadata.single_table import SingleTableMetadata
from ..sampling import Condition
from ..single_table.base import BaseSynthesizer
from ..single_table.base import BaseSingleTableSynthesizer
from ..single_table.ctgan import LossValuesMixin
from rdt.transformers import FloatFormatter
from ..data_processing.data_processor import DataProcessor
from ..metadata.single_table import SingleTableMetadata

from dataclasses import asdict, dataclass
from enum import Enum
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import OrderedDict
from typing import cast, Iterable, List, Optional, Tuple, Union
import abc
from category_encoders import BinaryEncoder, OneHotEncoder
from scipy.stats import mode
import math

from collections import Counter
from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple, Union

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class OutputType(Enum):
    """Supported variables types.

    Determines internal representation of variables and output layers in
    generation network.
    """

    DISCRETE = 0
    CONTINUOUS = 1

class Output(abc.ABC):
    """Stores metadata for a variable, used for both features and attributes."""

    def __init__(self, name: str):
        self.name = name

        self.is_fit = False

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of the transformed data produced for this variable."""
        ...

    def fit(self, column: np.ndarray):
        """Fit metadata and encoder params to data.

        Args:
            column: 1-d numpy array
        """
        if len(column.shape) != 1:
            raise ValueError("Expected 1-d numpy array for fit()")

        self._fit(column)
        self.is_fit = True

    def transform(self, column: np.ndarray) -> np.ndarray:
        """Transform data to internal representation.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array
        """
        if len(column.shape) != 1:
            raise ValueError("Expected 1-d numpy array for transform()")

        if not self.is_fit:
            raise RuntimeError("Cannot transform before output is fit()")
        else:
            return self._transform(column)

    def inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Inverse transform from internal representation to original data space.

        Args:
            columns: 2-d numpy array

        Returns:
            1-d numpy array in original data space
        """
        if not self.is_fit:
            raise RuntimeError("Cannot inverse transform before output is fit()")
        else:
            return self._inverse_transform(columns)

    @abc.abstractmethod
    def _fit(self, column: np.ndarray): ...

    @abc.abstractmethod
    def _transform(self, columns: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray: ...
    
class Normalization(Enum):
    """Normalization types for continuous variables.

    Determines if a sigmoid (ZERO_ONE) or tanh (MINUSONE_ONE) activation is used
    for the output layers in the generation network.
    """

    ZERO_ONE = 0
    MINUSONE_ONE = 1
    
class DfStyle(str, Enum):
    """Supported styles for parsing pandas DataFrames.

    See `train_dataframe` method in dgan.py for details.
    """

    WIDE = "wide"
    LONG = "long"
    
def rescale(
    original: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    """Scale continuous variable to [0,1] or [-1,1].

    Args:
        original: data in original space
        normalization: output range for scaling, ZERO_ONE or MINUSONE_ONE
        global_min: minimum to use for scaling, either a scalar or has same
            shape as original
        global_max: maximum to use for scaling, either a scalar or has same
            shape as original

    Returns:
        Data in transformed space
    """

    range = np.maximum(global_max - global_min, 1e-6)
    if normalization == Normalization.ZERO_ONE:
        return (original - global_min) / range
    elif normalization == Normalization.MINUSONE_ONE:
        return (2.0 * (original - global_min) / range) - 1.0

def rescale_inverse(
    transformed: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    """Invert continuous scaling to map back to original space.

    Args:
        transformed: data in transformed space
        normalization: output range for scaling, ZERO_ONE or MINUSONE_ONE
        global_min: minimum to use for scaling, either a scalar or has same
            dimension as original.shape[0] for scaling each time series
            independently
        global_max: maximum to use for scaling, either a scalar or has same
            dimension as original.shape[0]

    Returns:
        Data in original space
    """
    range = global_max - global_min
    if normalization == Normalization.ZERO_ONE:
        return transformed * range + global_min
    elif normalization == Normalization.MINUSONE_ONE:
        return ((transformed + 1.0) / 2.0) * range + global_min

class OneHotEncodedOutput(Output):
    """Metadata for a one-hot encoded variable."""

    def __init__(self, name: str, dim=None):
        """
        Args:
            name: name of variable
            dim: use to directly setup encoder for [0,1,2,,...,dim-1] values, if
                not None, calling fit() is not required. Provided for easier
                backwards compatability. Preferred usage is dim=None and then
                call fit() on the instance.
        """
        super().__init__(name)

        if dim is not None:
            self.fit(np.arange(dim))

    @property
    def dim(self) -> int:
        """Dimension of the transformed data produced by one-hot encoding."""
        if self.is_fit:
            return len(self._encoder.get_feature_names())
        else:
            raise RuntimeError("Cannot return dim before output is fit()")

    def _fit(self, column: np.ndarray):
        """Fit one-hot encoder.

        Args:
            column: 1-d numpy array
        """
        # Use cols=0 to always do the encoding, even if the input is integer or
        # float.
        self._encoder = OneHotEncoder(cols=0, return_df=False)

        self._encoder.fit(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply one-hot encoding.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array of encoded data
        """
        return self._encoder.transform(column).astype("float", casting="safe")

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert one-hot encoding.

        Args:
            columns: 2-d numpy array of floats or integers

        Returns:
            1-d numpy array
        """
        if len(columns.shape) != 2:
            raise ValueError(
                f"Expected 2-d numpy array, received shape={columns.shape}"
            )
        # Category encoders only inverts exact match binary rows, so need to do
        # argmax and then convert back to full binary matrix.
        # Might be more efficient to eventually do everything ourselves and not
        # use OneHotEncoder.
        indices = np.argmax(columns, axis=1)
        b = np.zeros(columns.shape)
        b[np.arange(len(indices)), indices] = 1

        return self._encoder.inverse_transform(b).flatten()
class BinaryEncodedOutput(Output):
    """Metadata for a binary encoded variable."""

    def __init__(self, name: str, dim=None):
        """
        Args:
            name: name of variable
            dim: use to directly setup encoder for [0,1,2,,...,dim-1] values, if
                not None, calling fit() is not required. Provided for easier
                backwards compatability. Preferred usage is dim=None and then
                call fit() on the instance.
        """
        super().__init__(name)

        self._convert_to_int = False

        if dim is not None:
            self.fit(np.arange(dim))

    @property
    def dim(self) -> int:
        """Dimension of the transformed data produced by binary encoding."""
        if self.is_fit:
            return len(self._encoder.get_feature_names())
        else:
            raise RuntimeError("Cannot return dim before output is fit()")

    def _fit(self, column: np.ndarray):
        """Fit binary encoder.


        Args:
            column: 1-d numpy array
        """
        # Use cols=0 to always do the encoding, even if the input is integer or
        # float.
        self._encoder = BinaryEncoder(cols=0, return_df=False)

        if type(column) != np.array:
            column = np.array(column)
        else:
            column = column.copy()

        # BinaryEncoder fails a lot if the input is integer (tries to cast to
        # int during inverse transform, but often have NaNs). So force any
        # numeric column to float.
        if np.issubdtype(column.dtype, np.integer):
            column = column.astype("float")
            self._convert_to_int = True

        # Use proxy value for nans if present so we can decode them explicitly
        # and differentiate from decoding failures.
        nan_mask = [x is np.nan for x in column]
        if np.sum(nan_mask) > 0:
            self._nan_proxy = uuid.uuid4().hex
            # Always make a copy at beginning of this function, so in place
            # change is okay.
            column[nan_mask] = self._nan_proxy
        else:
            self._nan_proxy = None

        # Store mode to use for unmapped binary codes.
        self._mode = mode(column).mode[0]

        self._encoder.fit(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply binary encoding.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array of encoded data
        """
        column = column.copy()
        if self._nan_proxy:
            nan_mask = [x is np.nan for x in column]
            column[nan_mask] = self._nan_proxy

        return self._encoder.transform(column).astype("float", casting="safe")

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert binary encoding.

        Args:
            columns: 2-d numpy array of floats or integers

        Returns:
            1-d numpy array
        """
        if len(columns.shape) != 2:
            raise ValueError(
                f"Expected 2-d numpy array, received shape={columns.shape}"
            )

        # Threshold to binary matrix
        binary = (columns > 0.5).astype("int")

        original_data = self._encoder.inverse_transform(binary).flatten()

        nan_mask = [x is np.nan for x in original_data]

        original_data[nan_mask] = self._mode

        # Now that decoding failure nans are replaced with the mode, replace
        # nan_proxy values with nans.
        if self._nan_proxy:
            nan_proxy_mask = [x == self._nan_proxy for x in original_data]
            original_data[nan_proxy_mask] = np.nan

        if self._convert_to_int:
            # TODO: store original type for conversion?
            original_data = original_data.astype("int")

        return original_data
class ContinuousOutput(Output):
    """Metadata for continuous variables."""

    def __init__(
        self,
        name: str,
        normalization: Normalization,
        apply_feature_scaling: bool,
        apply_example_scaling: bool,
        *,
        global_min: Optional[float] = None,
        global_max: Optional[float] = None,
    ):
        """
        Args:
            name: name of variable
            normalization: range of transformed value
            apply_feature_scaling: should values be scaled
            apply_example_scaling: should per-example scaling be used
            global_min: backwards compatability to set range in constructor,
                preferred to use fit()
            global_max: backwards compatability to set range in constructor
        """
        super().__init__(name)

        self.normalization = normalization

        self.apply_feature_scaling = apply_feature_scaling
        self.apply_example_scaling = apply_example_scaling

        if (global_min is None) != (global_max is None):
            raise ValueError("Must provide both global_min and global_max")

        if global_min is not None:
            self.is_fit = True
            self.global_min = global_min
            self.global_max = global_max

    @property
    def dim(self) -> int:
        """Dimension of transformed data."""
        return 1

    def _fit(self, column):
        """Fit continuous variable encoding/scaling.

        Args:
            column: 1-d numpy array
        """
        column = column.astype("float")
        self.global_min = np.nanmin(column)
        self.global_max = np.nanmax(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply continuous variable encoding/scaling.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array of rescaled data
        """
        column = column.astype("float")

        if self.apply_feature_scaling:
            return rescale(
                column, self.normalization, self.global_min, self.global_max
            ).reshape((-1, 1))
        else:
            return column.reshape((-1, 1))

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert continus variable encoding/scaling.

        Args:
            columns: numpy array

        Returns:
            numpy array
        """
        if self.apply_feature_scaling:
            return rescale_inverse(
                columns, self.normalization, self.global_min, self.global_max
            ).flatten()
        else:
            return columns.flatten()
    
def validation_check(
    features: List[np.ndarray],
    continuous_features_ind: List[int],
    invalid_examples_ratio_cutoff: float = 0.5,
    nans_ratio_cutoff: float = 0.1,
    consecutive_nans_max: int = 5,
    consecutive_nans_ratio_cutoff: float = 0.05,
) -> np.ndarray:
    """Checks if continuous features of examples are valid.

    Returns a 1-d numpy array of booleans with shape (#examples) indicating
    valid examples.
    Examples with continuous features fall into 3 categories: good, valid (fixable) and
    invalid (non-fixable).
    - "Good" examples have no NaNs.
    - "Valid" examples have a low percentage of nans and a below a threshold number of
    consecutive NaNs.
    - "Invalid" are the rest, and are marked "False" in the returned array.  Later on,
    these are omitted from training. If there are too many, later, we error out.

    Args:
        features: list of 2-d numpy arrays, each element is a sequence of
            possibly varying length
        continuous_features_ind: list of indices of continuous features to
            analyze, indexes the 2nd dimension of the sequence arrays in
            features
        invalid_examples_ratio_cutoff: Error out if the invalid examples ratio
            in the dataset is higher than this value.
        nans_ratio_cutoff: If the percentage of nans for any continuous feature
           in an example is greater than this value, the example is invalid.
        consecutive_nans_max: If the maximum number of consecutive nans in a
           continuous feature is greater than this number, then that example is
           invalid.
        consecutive_nans_ratio_cutoff: If the maximum number of consecutive nans
            in a continuous feature is greater than this ratio times the length of
            the example (number samples), then the example is invalid.

    Returns:
        valid_examples: 1-d numpy array of booleans indicating valid examples with
        shape (#examples).

    """
    # Check for the nans ratio per examples and feature.
    # nan_ratio_feature is a 2-d numpy array of size (#examples,#features)
    nan_ratio_feature = np.array(
        [
            [
                np.mean(np.isnan(seq[:, ind].astype("float")))
                for ind in continuous_features_ind
            ]
            for seq in features
        ]
    )

    nan_ratio = nan_ratio_feature < nans_ratio_cutoff

    # Check for max number of consecutive NaN values per example and feature.
    # cons_nans_feature is a 2-d numpy array of size (#examples,#features)
    cons_nans_feature = np.array(
        [
            [
                find_max_consecutive_nans(seq[:, ind].astype("float"))
                for ind in continuous_features_ind
            ]
            for seq in features
        ]
    )
    # With examples of variable sequence length, the threshold for allowable
    # consecutive nans may be different for each example.
    cons_nans_threshold = np.clip(
        [consecutive_nans_ratio_cutoff * seq.shape[0] for seq in features],
        a_min=2,
        a_max=consecutive_nans_max,
    ).reshape((-1, 1))
    cons_nans = cons_nans_feature < cons_nans_threshold

    # The two above checks should pass for a valid example for all features, otherwise
    # the example is invalid.
    valid_examples_per_feature = np.logical_and(nan_ratio, cons_nans)
    valid_examples = np.all(valid_examples_per_feature, axis=1)

    if np.mean(valid_examples) < invalid_examples_ratio_cutoff:
        raise ValueError(
            f"More than {100*invalid_examples_ratio_cutoff}% invalid examples in the continuous features. Please reduce the ratio of the NaNs and try again!"  # noqa
        )

    return valid_examples

def nan_linear_interpolation(
    features: List[np.ndarray], continuous_features_ind: List[int]
):
    """Replaces all NaNs via linear interpolation.

    Changes numpy arrays in features in place.

    Args:
        features: list of 2-d numpy arrays, each element is a sequence of shape
            (sequence_len, #features)
        continuous_features_ind: features to apply nan interpolation to, indexes
            the 2nd dimension of the sequence arrays of features
    """
    for seq in features:
        for ind in continuous_features_ind:
            continuous_feature = seq[:, ind].astype("float")
            is_nan = np.isnan(continuous_feature)
            if is_nan.any():
                ind_func = lambda z: z.nonzero()[0]  # noqa
                seq[is_nan, ind] = np.interp(
                    ind_func(is_nan), ind_func(~is_nan), continuous_feature[~is_nan]
                )

def transform_attributes(
    original_data: np.ndarray,
    outputs: List[Output],
) -> np.ndarray:
    """Transform attributes to internal representation expected by DGAN.

    See transform_features pydoc for more details on how the original_data is
    changed.

    Args:
        original_data: data to transform as a 2d numpy array
        outputs: Output metadata for each attribute

    Returns:
        2d numpy array of the internal representation of data.
    """
    parts = []
    for index, output in enumerate(outputs):
        parts.append(output.transform(original_data[:, index]))

    return np.concatenate(parts, axis=1, dtype=np.float32)

def _grouped_min_and_max(
    example_ids: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min and max for each example.

    Sorts by example_ids, then values, and then indexes into the sorted values
    to efficiently obtain min/max. Compute both min and max in one function to
    reuse the sorted objects.

    Args:
        example_ids: 1d numpy array of example ids, mapping each element
            in values to an example/sequence
        values: 1d numpy array

    Returns:
        Tuple of min and max values for each example/sequence, each is a 1d
        numpy array of size # of unique example_ids. The min and max values are
        for the sorted example_ids, so the first element is the min/max of the
        smallest example_id value, and so on.
    """
    # lexsort primary key is last element, so sorts by example_ids first, then
    # values
    order = np.lexsort((values, example_ids))
    g = example_ids[order]
    d = values[order]
    # Construct index marking lower borders between examples to capture the min
    # values
    min_index = np.empty(len(g), dtype="bool")
    min_index[0] = True
    min_index[1:] = g[1:] != g[:-1]
    # Construct index marking upper borders between groups to capture the max
    # values
    max_index = np.empty(len(g), dtype="bool")
    max_index[-1] = True
    max_index[:-1] = g[1:] != g[:-1]

    return d[min_index], d[max_index]
def transform_features(
    original_data: List[np.ndarray],
    outputs: List[Output],
    max_sequence_len: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Transform features to internal representation expected by DGAN.

    Specifically, performs the following changes:

    * Converts discrete variables to one-hot encoding
    * Scales continuous variables by feature or example min/max to [0,1] or
        [-1,1]
    * Create per example attributes with midpoint and half-range when
        apply_example_scaling is True

    Args:
        original_data: data to transform as a list of 2d numpy
            arrays, each element is a sequence
        outputs: Output metadata for each variable
        max_sequence_len: pad all sequences to max_sequence_len

    Returns:
        Internal representation of data. A tuple of 3d numpy array of features
        and optional 2d numpy array of additional_attributes.
    """
    sequence_lengths = [seq.shape[0] for seq in original_data]
    if max(sequence_lengths) > max_sequence_len:
        raise ValueError(
            f"Found sequence with length {max(sequence_lengths)}, longer than max_sequence_len={max_sequence_len}"
        )
    example_ids = np.repeat(range(len(original_data)), sequence_lengths)

    long_data = np.vstack(original_data)

    parts = []
    additional_attribute_parts = []
    for index, output in enumerate(outputs):
        # NOTE: isinstance(output, DiscreteOutput) does not work consistently
        #       with all import styles in jupyter notebooks, using string
        #       comparison instead.
        if "OneHotEncodedOutput" in str(
            output.__class__
        ) or "BinaryEncodedOutput" in str(output.__class__):
            transformed_data = output.transform(long_data[:, index])
            parts.append(transformed_data)
        elif "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)

            raw = long_data[:, index]

            feature_scaled = output.transform(raw)

            if output.apply_example_scaling:
                # Group-wise mins and maxes, dimension of each is (# examples,)
                group_mins, group_maxes = _grouped_min_and_max(
                    example_ids, feature_scaled.flatten()
                )
                # Project back to size of long data
                mins = np.repeat(group_mins, sequence_lengths).reshape((-1, 1))
                maxes = np.repeat(group_maxes, sequence_lengths).reshape((-1, 1))

                additional_attribute_parts.append(
                    ((group_mins + group_maxes) / 2).reshape((-1, 1))
                )
                additional_attribute_parts.append(
                    ((group_maxes - group_mins) / 2).reshape((-1, 1))
                )

                scaled = rescale(feature_scaled, output.normalization, mins, maxes)
            else:
                scaled = feature_scaled

            parts.append(scaled.reshape(-1, 1))
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    long_transformed = np.concatenate(parts, axis=1, dtype=np.float32)

    # Fit possibly jagged sequences into 3d numpy array. Pads shorter sequences
    # with all 0s in the internal representation.
    features_transformed = np.zeros(
        (len(original_data), max_sequence_len, long_transformed.shape[1]),
        dtype=np.float32,
    )
    i = 0
    for example_index, length in enumerate(sequence_lengths):
        features_transformed[example_index, 0:length, :] = long_transformed[
            i : (i + length), :
        ]
        i += length

    additional_attributes = None
    if additional_attribute_parts:
        additional_attributes = np.concatenate(
            additional_attribute_parts, axis=1, dtype=np.float32
        )

    return features_transformed, additional_attributes


def inverse_transform_attributes(
    transformed_data: np.ndarray,
    outputs: List[Output],
) -> Optional[np.ndarray]:
    """Inverse of transform_attributes to map back to original space.

    Args:
        transformed_data: 2d numpy array of internal representation
        outputs: Output metadata for each variable
    """
    # TODO: we should not use nans as an indicator and just not call this
    # method, or use a zero sized numpy array, to indicate no attributes.
    if np.isnan(transformed_data).any():
        return None
    parts = []
    transformed_index = 0
    for output in outputs:
        original = output.inverse_transform(
            transformed_data[:, transformed_index : (transformed_index + output.dim)]
        )
        parts.append(original.reshape((-1, 1)))
        transformed_index += output.dim

    return np.hstack(parts)


def inverse_transform_features(
    transformed_data: np.ndarray,
    outputs: List[Output],
    additional_attributes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse of transform_features to map back to original space.

    Args:
        transformed_data: 3d numpy array of internal representation data
        outputs: Output metadata for each variable
        additional_attributes: midpoint and half-ranges for outputs with
            apply_example_scaling=True

    Returns:
        List of numpy arrays, each element corresponds to one sequence with 2d
        array of (time x variables).
    """
    transformed_index = 0
    additional_attribute_index = 0
    parts = []
    for output in outputs:
        if "OneHotEncodedOutput" in str(
            output.__class__
        ) or "BinaryEncodedOutput" in str(output.__class__):

            v = transformed_data[
                :, :, transformed_index : (transformed_index + output.dim)
            ]
            target_shape = (transformed_data.shape[0], transformed_data.shape[1], 1)

            original = output.inverse_transform(v.reshape((-1, v.shape[-1])))

            parts.append(original.reshape(target_shape))
            transformed_index += output.dim
        elif "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)

            transformed = transformed_data[:, :, transformed_index]

            if output.apply_example_scaling:
                if additional_attributes is None:
                    raise ValueError(
                        "Must provide additional_attributes if apply_example_scaling=True"
                    )

                midpoint = additional_attributes[:, additional_attribute_index]
                half_range = additional_attributes[:, additional_attribute_index + 1]
                additional_attribute_index += 2

                mins = midpoint - half_range
                maxes = midpoint + half_range
                mins = np.expand_dims(mins, 1)
                maxes = np.expand_dims(maxes, 1)

                example_scaled = rescale_inverse(
                    transformed,
                    normalization=output.normalization,
                    global_min=mins,
                    global_max=maxes,
                )
            else:
                example_scaled = transformed

            original = output.inverse_transform(example_scaled)

            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            original = original.reshape(target_shape)

            parts.append(original)
            transformed_index += 1
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    return np.concatenate(parts, axis=2)

class _DataFrameConverter(abc.ABC):
    """Abstract class for converting DGAN input to and from a DataFrame."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Class name used for serialization."""
        ...

    @property
    @abc.abstractmethod
    def attribute_types(self) -> List[OutputType]:
        """Output types used for attributes."""
        ...

    @property
    @abc.abstractmethod
    def feature_types(self) -> List[OutputType]:
        """Output types used for features."""
        ...

    @abc.abstractmethod
    def convert(self, df: pd.DataFrame):
        """Convert DataFrame to DGAN input format.

        Args:
            df: DataFrame of training data

        Returns:
            Attribute (optional) and feature numpy arrays.
        """
        ...

    @abc.abstractmethod
    def invert(
        self,
        attributes: Optional[np.ndarray],
        features: List[np.ndarray],
    ) -> pd.DataFrame:
        """Invert from DGAN input format back to DataFrame.

        Args:
            attributes: 2d numpy array of attributes
            features: list of 2d numpy arrays

        Returns:
            DataFrame representing attributes and features in original format.
        """
        ...

    def state_dict(self) -> Dict:
        """Dictionary describing this converter to use in saving and loading."""
        state = self._state_dict()
        state["name"] = self.name
        return state

    @abc.abstractmethod
    def _state_dict(self) -> Dict:
        """Subclass specific dictionary for saving and loading."""
        ...

    @classmethod
    def load_from_state_dict(cls, state: Dict):
        """Load a converter previously saved to a state dictionary."""
        # Assumes saved state was created with `state_dict()` method with name
        # and other params to initialize the class specified in
        # CONVERTER_CLASS_MAP. Care is required when modifying constructor
        # params or changing names if backwards compatibility is required.
        sub_class = CONVERTER_CLASS_MAP[state.pop("name")]

        return sub_class(**state)

def _discrete_cols_to_int(
    df: pd.DataFrame, discrete_columns: Optional[List[str]]
) -> pd.DataFrame:
    # Convert discrete columns to int where possible.
    if discrete_columns is None:
        return df

    missing_discrete = set()
    for c in discrete_columns:
        try:
            df[c] = df[c].astype("int")
        except ValueError:
            continue
        except KeyError:
            missing_discrete.add(c)

    if missing_discrete:
        ValueError(
            f"The following discrete columns ({missing_discrete}) were not in the generated DataFrame, you may want to ensure this is intended!"  # noqa
        )

    return df

class _WideDataFrameConverter(_DataFrameConverter):
    """Convert "wide" format DataFrames.

    Expects one row for each example with 0 or more attribute columns and 1
    column per time point in the time series.
    """

    def __init__(
        self,
        attribute_columns: List[str],
        feature_columns: List[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        feature_types: List[OutputType],
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._feature_columns = feature_columns
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._feature_types = feature_types

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        attribute_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Create a converter instance.

        See `train_dataframe` for parameter details.
        """
        if attribute_columns is None:
            attribute_columns = []
        else:
            attribute_columns = attribute_columns

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in attribute_columns]
        else:
            feature_columns = feature_columns

        df_column_order = [
            c for c in df.columns if c in attribute_columns or c in feature_columns
        ]

        if discrete_columns is None:
            discrete_column_set = set()
        else:
            discrete_column_set = set(discrete_columns)

        # Check for string columns and ensure they are considered discrete.
        for column_name in df.columns:
            if df[column_name].dtype == "O":
                logging.info(
                    f"Marking column {column_name} as discrete because its type is string/object."
                )
                discrete_column_set.add(column_name)

        attribute_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in attribute_columns
        ]
        # With wide format, there's always 1 feature. It's only discrete if
        # every column used (every time point) is discrete.
        if all(c in discrete_column_set for c in feature_columns):
            feature_types = [OutputType.DISCRETE]
        else:
            feature_types = [OutputType.CONTINUOUS]

        return _WideDataFrameConverter(
            attribute_columns=attribute_columns,
            feature_columns=feature_columns,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            feature_types=feature_types,
        )

    @property
    def name(self) -> str:
        return "WideDataFrameConverter"

    @property
    def attribute_types(self):
        return self._attribute_types

    @property
    def feature_types(self):
        return self._feature_types

    def convert(self, df: pd.DataFrame):
        if self._attribute_columns:
            attributes = df[self._attribute_columns].to_numpy()
        else:
            attributes = None

        features = np.expand_dims(df[self._feature_columns].to_numpy(), axis=-1)

        return attributes, [seq for seq in features], [seq.shape[0] for seq in features]

    def invert(
        self, attributes: Optional[np.ndarray], features: List[np.ndarray]
    ) -> pd.DataFrame:
        if self._attribute_columns:
            if attributes is None:
                raise RuntimeError(
                    "Data converter with attribute columns expects attributes array, received None"
                )
            data = np.concatenate(
                (attributes, np.vstack([seq.reshape((1, -1)) for seq in features])),
                axis=1,
            )
        else:
            data = np.vstack([seq.reshape((1, -1)) for seq in features])

        df = pd.DataFrame(data, columns=self._attribute_columns + self._feature_columns)

        # Convert discrete columns to int where possible.
        df = _discrete_cols_to_int(df, self._discrete_columns)

        # Ensure we match the original ordering
        return df[self._df_column_order]

    def _state_dict(self) -> Dict:
        return {
            "attribute_columns": self._attribute_columns,
            "feature_columns": self._feature_columns,
            "discrete_columns": self._discrete_columns,
            "df_column_order": self._df_column_order,
            "attribute_types": self._attribute_types,
            "feature_types": self._feature_types,
        }

def _add_generation_flag(
    sequence: np.ndarray, generation_flag_index: int
) -> np.ndarray:
    """Adds column indicating continuing and end time points in sequence.

    Args:
        sequence: 2-d numpy array of a single sequence
        generation_flag_index: index of column to insert

    Returns:
        New array including the generation flag column
    """
    # Generation flag is all True
    flag_column = np.full((sequence.shape[0], 1), True)
    # except last value is False to indicate the end of the sequence
    flag_column[-1, 0] = False

    return np.hstack(
        (
            sequence[:, :generation_flag_index],
            flag_column,
            sequence[:, generation_flag_index:],
        )
    )

class _LongDataFrameConverter(_DataFrameConverter):
    """Convert "long" format DataFrames.

    Expects one row per time point. Splits into examples based on specified
    example id column.
    """

    def __init__(
        self,
        attribute_columns: List[str],
        feature_columns: List[str],
        example_id_column: Optional[str],
        time_column: Optional[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        feature_types: List[OutputType],
        time_column_values: Optional[List[str]],
        generation_flag_index: Optional[int] = None,
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._feature_columns = feature_columns
        self._example_id_column = example_id_column
        self._time_column = time_column
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._feature_types = feature_types
        self._time_column_values = time_column_values
        self._generation_flag_index = generation_flag_index

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        max_sequence_len: int,
        attribute_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        example_id_column: Optional[str] = None,
        time_column: Optional[str] = None,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Create a converter instance.

        See `train_dataframe` for parameter details.
        """
        if attribute_columns is None:
            attribute_columns = []
        else:
            attribute_columns = attribute_columns

        given_columns = set(attribute_columns)
        if example_id_column is not None:
            given_columns.add(example_id_column)
        if time_column is not None:
            given_columns.add(time_column)

        if feature_columns is None:
            # If not specified, use remaining columns in the data frame that
            # are not used elsewhere
            feature_columns = [c for c in df.columns if c not in given_columns]
        else:
            feature_columns = feature_columns

        # Add feature columns too, so given_columns contains all columns of df
        # that we are actually using
        given_columns.update(feature_columns)

        df_column_order = [c for c in df.columns if c in given_columns]

        if discrete_columns is None:
            discrete_column_set = set()
        else:
            discrete_column_set = set(discrete_columns)

        # Check for string columns and ensure they are considered discrete.
        for column_name in df.columns:
            # Check all columns being used, except time_column and
            # example_id_column which are not directly modeled.
            if (
                df[column_name].dtype == "O"
                and column_name in given_columns
                and column_name != time_column
                and column_name != example_id_column
            ):
                logging.info(
                    f"Marking column {column_name} as discrete because its type is string/object."
                )
                discrete_column_set.add(column_name)

        attribute_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in attribute_columns
        ]
        feature_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in feature_columns
        ]

        if time_column:
            if example_id_column:
                # Assume all examples are for the same time points, e.g., always
                # from 2020 even if df has examples from different years.
                df_time_example = df[[time_column, example_id_column]]
                # Use first example grouping (iloc[0]), then grab the time
                # column values used by that example from the numpy array
                # ([:,0]).
                time_values = (
                    df_time_example.groupby(example_id_column)
                    .apply(pd.DataFrame.to_numpy)
                    .iloc[0][:, 0]
                )

                time_column_values = list(sorted(time_values))
            else:
                time_column_values = list(sorted(df[time_column]))
        else:
            time_column_values = None

        # generation_flag_index is the index in feature_types (and thus
        # features) of the boolean variable indicating the end of sequence.
        # generation_flag_index=None means there are no variable length
        # sequences, so the indicator variable is not needed and no boolean
        # feature is added.
        generation_flag_index = None
        if example_id_column:
            id_counter = Counter(df[example_id_column])
            has_variable_length_sequences = False
            for item in id_counter.most_common():
                if item[1] > max_sequence_len:
                    raise ValueError(
                        f"Found sequence with length {item[1]}, longer than max_sequence_len={max_sequence_len}"
                    )
                elif item[1] < max_sequence_len:
                    has_variable_length_sequences = True

            if has_variable_length_sequences:
                generation_flag_index = len(feature_types)
                feature_types.append(OutputType.DISCRETE)

        return cls(
            attribute_columns=attribute_columns,
            feature_columns=feature_columns,
            example_id_column=example_id_column,
            time_column=time_column,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            feature_types=feature_types,
            time_column_values=time_column_values,
            generation_flag_index=generation_flag_index,
        )

    @property
    def name(self) -> str:
        return "LongDataFrameConverter"

    @property
    def attribute_types(self):
        return self._attribute_types

    @property
    def feature_types(self):
        return self._feature_types

    def convert(self, df: pd.DataFrame):

        if self._time_column is not None:
            sorted_df = df.sort_values(by=[self._time_column])
        else:
            sorted_df = df

        if self._example_id_column is not None:
            # Use example_id_column to split into separate time series
            df_features = sorted_df[self._feature_columns]

            features = list(
                df_features.groupby(sorted_df[self._example_id_column]).apply(
                    pd.DataFrame.to_numpy
                )
            )

            if self._attribute_columns:
                df_attributes = sorted_df[
                    self._attribute_columns + [self._example_id_column]
                ]

                # Check that attributes are the same for all rows with the same
                # example id. Use custom min and max functions that ignore nans.
                # Using pandas min() and max() functions leads to errors when a
                # single example has a mix of string and nan values for an
                # attribute across different rows because str and float are not
                # comparable.
                def custom_min(a):
                    return min((x for x in a if x is not np.nan), default=np.nan)

                def custom_max(a):
                    return max((x for x in a if x is not np.nan), default=np.nan)

                attribute_mins = df_attributes.groupby(self._example_id_column).apply(
                    lambda frame: frame.apply(custom_min)
                )
                attribute_maxes = df_attributes.groupby(self._example_id_column).apply(
                    lambda frame: frame.apply(custom_max)
                )

                for column in self._attribute_columns:
                    # Use custom list comprehension for the comparison to allow
                    # nan attribute values (nans don't compare equal so any
                    # example with an attribute of nan would fail the min/max
                    # equality check).
                    comparison = [
                        x is np.nan if y is np.nan else x == y
                        for x, y in zip(attribute_mins[column], attribute_maxes[column])
                    ]
                    if not np.all(comparison):
                        raise ValueError(
                            f"Attribute {column} is not constant within each example."
                        )

                attributes = (
                    df_attributes.groupby(self._example_id_column).min().to_numpy()
                )
            else:
                attributes = None
        else:
            # No example_id column provided to create multiple examples, so we
            # create one example from all time points.
            features = [sorted_df[self._feature_columns].to_numpy()]

            # Check that attributes are the same for all rows (since they are
            # all implicitly in the same example)
            for column in self._attribute_columns:
                if sorted_df[column].nunique() != 1:
                    raise ValueError(f"Attribute {column} is not constant for all rows.")

            if self._attribute_columns:
                # With one example, attributes should all be constant, so grab from
                # the first row. Need to add first (example) dimension.
                attributes = np.expand_dims(
                    sorted_df[self._attribute_columns].iloc[0, :].to_numpy(), axis=0
                )
            else:
                attributes = None

        if self._generation_flag_index is not None:
            features = [
                _add_generation_flag(seq, self._generation_flag_index)
                for seq in features
            ]
        return attributes, features, [int(seq.shape[0]) for seq in features]

    def invert(
        self,
        attributes: Optional[np.ndarray],
        features: List[np.ndarray],
    ) -> pd.DataFrame:
        sequences = []
        for seq_index, seq in enumerate(features):
            if self._generation_flag_index is not None:
                # Remove generation flag and truncate sequences based on the values.
                # The first value of False in the generation flag indicates the last
                # time point.
                try:
                    first_false = np.min(
                        np.argwhere(seq[:, self._generation_flag_index] == False)
                    )
                    # Include the time point with the first False in generation
                    # flag
                    seq = seq[: (first_false + 1), :]
                except ValueError:
                    # No False found in generation flag column, use all time
                    # points
                    pass

                # Remove the generation flag column
                seq = np.delete(seq, self._generation_flag_index, axis=1)

            if seq.shape[1] != len(self._feature_columns):
                raise RuntimeError(
                    "Unable to invert features back to data frame, "
                    + f"converter expected {len(self._feature_columns)} features, "
                    + f"received numpy array with {seq.shape[1]}"
                )

            seq_column_parts = [seq]
            if self._attribute_columns:
                if attributes is None:
                    raise RuntimeError(
                        "Data converter with attribute columns expects attributes array, received None"
                    )
                seq_attributes = np.repeat(
                    attributes[seq_index : (seq_index + 1), :], seq.shape[0], axis=0
                )
                seq_column_parts.append(seq_attributes)

            if self._example_id_column:
                # TODO: match example_id style of original data somehow
                seq_column_parts.append(np.full((seq.shape[0], 1), seq_index))

            if self._time_column:
                if self._time_column_values is None:
                    raise RuntimeError(
                        "time_column is present, but not time_column_values"
                    )
                values = [
                    v
                    for _, v in zip(
                        range(seq.shape[0]), cycle(self._time_column_values)
                    )
                ]
                seq_column_parts.append(np.array(values).reshape((-1, 1)))

            sequences.append(np.hstack(seq_column_parts))

        column_names = self._feature_columns + self._attribute_columns

        if self._example_id_column:
            column_names.append(self._example_id_column)
        if self._time_column:
            column_names.append(self._time_column)

        df = pd.DataFrame(np.vstack(sequences), columns=column_names)

        for c in df.columns:
            try:
                df[c] = df[c].astype("float64")
            except ValueError:
                continue
            except TypeError:
                continue

        # Convert discrete columns to int where possible.
        df = _discrete_cols_to_int(
            df,
            (self._discrete_columns),
        )
        if self._example_id_column:
            df = _discrete_cols_to_int(df, [self._example_id_column])

        return df[self._df_column_order]

    def _state_dict(self) -> Dict:
        return {
            "attribute_columns": self._attribute_columns,
            "feature_columns": self._feature_columns,
            "example_id_column": self._example_id_column,
            "time_column": self._time_column,
            "df_column_order": self._df_column_order,
            "discrete_columns": self._discrete_columns,
            "attribute_types": self._attribute_types,
            "feature_types": self._feature_types,
            "time_column_values": self._time_column_values,
            "generation_flag_index": self._generation_flag_index,
        }

CONVERTER_CLASS_MAP = {
    "WideDataFrameConverter": _WideDataFrameConverter,
    "LongDataFrameConverter": _LongDataFrameConverter,
}

def find_max_consecutive_nans(array: np.ndarray) -> int:
    """
    Returns the maximum number of consecutive NaNs in an array.

    Args:
        array: 1-d numpy array of time series per example.

    Returns:
        max_cons_nan: The maximum number of consecutive NaNs in a times series array.

    """
    # The number of consecutive nans are listed based on the index difference between the non-null values.
    max_cons_nan = np.max(
        np.diff(np.concatenate(([-1], np.where(~np.isnan(array))[0], [len(array)]))) - 1
    )
    return max_cons_nan

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data

def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb

def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len

def random_generator(batch_size, input_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - input_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, input_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], input_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    return np.stack(Z_mb)

def NormMinMax(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val  # [3661, 24, 6]

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

        Args:
          - input: input time-series features. (L, N, X) = (24, ?, 6)
          - h3: (num_layers, N, H). [3, ?, 24]

        Returns:
          - H: embeddings
        """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size= input_dim, hidden_size= hidden_dim, num_layers= num_layers)
        # self.norm = nn.BatchNorm1d(opt.hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H

class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(input_size= hidden_dim, hidden_size= input_dim, num_layers= num_layers)

        #  self.norm = nn.BatchNorm1d(opt.input_dim)
        self.fc = nn.Linear( input_dim,  input_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde

class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """

    def __init__(self, hidden_dim, input_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size= input_dim, hidden_size= hidden_dim, num_layers= num_layers)
        #   self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        #  g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E

class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size= hidden_dim, hidden_size= hidden_dim, num_layers= num_layers)
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        #  s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S

class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size= hidden_dim, hidden_size= hidden_dim, num_layers= num_layers)
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat

def create_outputs_from_data(
    attributes: Optional[np.ndarray],
    features: List[np.ndarray],
    attribute_types: Optional[List[OutputType]],
    feature_types: Optional[List[OutputType]],
    normalization: Normalization,
    apply_feature_scaling: bool = False,
    apply_example_scaling: bool = False,
    binary_encoder_cutoff: int = 150,
):
    """Create output metadata from data.

    Args:
        attributes: 2d numpy array of attributes
        features: list of 2d numpy arrays, each element is one sequence
        attribute_types: variable type for each attribute, assumes continuous if None
        feature_types: variable type for each feature, assumes continuous if None
        normalization: internal representation for continuous variables, scale
            to [0,1] or [-1,1]
        apply_feature_scaling: scale continuous variables inside the model, if
            False inputs must already be scaled to [0,1] or [-1,1]
        apply_example_scaling: include midpoint and half-range as additional
            attributes for each feature and scale per example, improves
            performance when time series ranges are highly variable
        binary_encoder_cutoff: use binary encoder (instead of one hot encoder) for
            any column with more than this many unique values
    """
    attribute_outputs = None
    if attributes is not None:
        if attribute_types is None:
            attribute_types = [OutputType.CONTINUOUS] * attributes.shape[1]
        elif len(attribute_types) != attributes.shape[1]:
            raise RuntimeError(
                "attribute_types must be the same length as the 2nd (last) dimension of attributes"
            )
        attribute_types = cast(List[OutputType], attribute_types)
        attribute_outputs = [
            create_output(
                index,
                t,
                attributes[:, index],
                normalization=normalization,
                apply_feature_scaling=apply_feature_scaling,
                # Attributes can never be normalized per example since there's
                # only 1 value for each variable per example.
                apply_example_scaling=False,
                binary_encoder_cutoff=binary_encoder_cutoff,
            )
            for index, t in enumerate(attribute_types)
        ]

    if feature_types is None:
        feature_types = [OutputType.CONTINUOUS] * features[0].shape[1]
    elif len(feature_types) != features[0].shape[1]:
        raise RuntimeError(
            "feature_types must be the same length as the 3rd (last) dimemnsion of features"
        )
    feature_types = cast(List[OutputType], feature_types)

    feature_outputs = [
        create_output(
            index,
            t,
            np.hstack([seq[:, index] for seq in features]),
            normalization=normalization,
            apply_feature_scaling=apply_feature_scaling,
            apply_example_scaling=apply_example_scaling,
            binary_encoder_cutoff=binary_encoder_cutoff,
        )
        for index, t in enumerate(feature_types)
    ]

    return attribute_outputs, feature_outputs

def create_output(
    index: int,
    t: OutputType,
    data: np.ndarray,
    normalization: Normalization,
    apply_feature_scaling: bool,
    apply_example_scaling: bool,
    binary_encoder_cutoff: int,
):
    """Create a single output from data.

    Args:
        index: index of variable within attributes or features
        t: type of output
        data: 1-d numpy array of data just for this variable
        normalization: see documentation in create_outputs_from_data
        apply_feature_scaling: see documentation in create_outputs_from_data
        apply_example_scaling: see documentation in create_outputs_from_data
        binary_encoder_cutoff: see documentation in create_outputs_from_data

    Returns:
        Output metadata instance
    """

    if t == OutputType.CONTINUOUS:
        output = ContinuousOutput(
            name="a" + str(index),
            normalization=normalization,
            apply_feature_scaling=apply_feature_scaling,
            apply_example_scaling=apply_example_scaling,
        )

    elif t == OutputType.DISCRETE:
        if data.dtype == "float":
            unique_count = len(np.unique(data))
        else:
            # Convert to str to ensure all elements are comparable (so unique
            # works as expected). In particular, this converts nan to "nan"
            # which is comparable.
            unique_count = len(np.unique(data.astype("str")))

        if unique_count > binary_encoder_cutoff:
            output = BinaryEncodedOutput(name="a" + str(index))
        else:
            output = OneHotEncodedOutput(name="a" + str(index))

    else:
        raise RuntimeError(f"Unknown output type={t}")

    output.fit(data.flatten())

    return output

def create_additional_attribute_outputs(feature_outputs: List[Output]) -> List[Output]:
    """Create outputs for midpoint and half ranges.

    Returns list of additional attribute metadata. For each feature with
    apply_example_scaling=True, adds 2 attributes, one for the midpoint of the
    sequence and one for the half range.

    Args:
        feature_outputs: output metadata for features

    Returns:
        List of Output instances for additional attributes
    """
    additional_attribute_outputs = []
    for output in feature_outputs:
        if "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)
            if output.apply_example_scaling:
                # Assumes feature data is already normalized to [0,1] or
                # [-1,1] according to output.normalization before the
                # per-example midpoint and half-range are calculated. So no
                # normalization is needed for these variables.
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_midpoint",
                        normalization=output.normalization,
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                        # TODO: are min/max really needed here since we aren't
                        # doing any scaling, could add an IdentityOutput instead?
                        global_min=(
                            0.0
                            if output.normalization == Normalization.ZERO_ONE
                            else -1.0
                        ),
                        global_max=1.0,
                    )
                )
                # The half-range variable always uses ZERO_ONE normalization
                # because it should always be positive.
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_half_range",
                        normalization=Normalization.ZERO_ONE,
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                        global_min=0.0,
                        global_max=1.0,
                    )
                )

    return additional_attribute_outputs

class TimeGan():
    """TimeGAN Class
    """
    def __init__(self, max_sequence_len: int, attribute_noise_dim: int = 10, feature_noise_dim: int = 10, recursive_module= "gru", hidden_dim= 24, num_layers= 3,
                 metric_iteration=10, beta1= 0.9, learning_rate= 0.001, gamma= 1, encoder_loss_weight_s = 0.1,
                 normalization: Normalization = Normalization.ZERO_ONE, apply_feature_scaling: bool = True, apply_example_scaling: bool = True, binary_encoder_cutoff: int = 150,
                 encoder_loss_weight_0 = 10, generator_loss_weight = 100, generator_steps= 2, attribute_outputs: Optional[List[Output]] = None, feature_outputs: Optional[List[Output]] = None,
                 epochs= 500000, batch_size= 16, verbose= False, wandb= False, device= "cuda"):

        self.seed(42)

        self._max_sequence_len = max_sequence_len
        self._attribute_noise_dim = attribute_noise_dim
        self._feature_noise_dim = feature_noise_dim
        self._recursive_module = recursive_module
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._metric_iteration = metric_iteration
        self._beta1 = beta1
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._encoder_loss_weight_s = encoder_loss_weight_s
        self._encoder_loss_weight_0 = encoder_loss_weight_0
        self._generator_loss_weight = generator_loss_weight
        self._generator_steps = generator_steps

        self._normalization = normalization
        self._apply_feature_scaling = apply_feature_scaling
        self._apply_example_scaling = apply_example_scaling
        self._binary_encoder_cutoff = binary_encoder_cutoff

        self._epochs = epochs
        self._batch_size = batch_size

        self._verbose = verbose
        self._wandb = wandb

        if device:
            if isinstance(device, str):
                self.device = device
            else:
                self.device = "cuda"
        else:
            self.device = "cuda"

        # -- Misc attributes
        self.times = []
        self.total_steps = 0

        if feature_outputs is not None and attribute_outputs is not None:
            self._build(attribute_outputs, feature_outputs)
        elif feature_outputs is not None or attribute_outputs is not None:
            raise RuntimeError(
                "feature_outputs and attribute_ouputs must either both be given or both be None"
            )

        self.data_frame_converter = None
        self.is_built = False

    def _build(self,
        attribute_outputs: Optional[List[Output]],
        feature_outputs: List[Output]):

        self.attribute_outputs = attribute_outputs
        self.additional_attribute_outputs = create_additional_attribute_outputs(
            feature_outputs
        )
        self.feature_outputs = feature_outputs

        if self.attribute_outputs is None:
            self.attribute_outputs = []
        attribute_dim = sum(output.dim for output in self.attribute_outputs)

        if not self.additional_attribute_outputs:
            self.additional_attribute_outputs = []
        additional_attribute_dim = sum(
            output.dim for output in self.additional_attribute_outputs
        )
        feature_dim = sum(output.dim for output in feature_outputs)

        #input_dim = attribute_dim + additional_attribute_dim + feature_dim
        input_dim = attribute_dim + feature_dim + additional_attribute_dim

        # Create and initialize networks.
        self.nete = Encoder(hidden_dim=self._hidden_dim, input_dim=input_dim, num_layers=self._num_layers).to(self.device)
        self.netr = Recovery(hidden_dim=self._hidden_dim, input_dim=input_dim, num_layers=self._num_layers).to(self.device)
        self.netg = Generator(hidden_dim=self._hidden_dim, input_dim=self._feature_noise_dim + self._attribute_noise_dim, num_layers=self._num_layers).to(self.device)
        self.netd = Discriminator(hidden_dim=self._hidden_dim, num_layers=self._num_layers).to(self.device)
        self.nets = Supervisor(hidden_dim=self._hidden_dim, num_layers=self._num_layers).to(self.device)

        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self._learning_rate, betas=(self._beta1, 0.999))

        self.is_built = True
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def forward_e(self, X):
        """ Forward propagate through netE
        """
        H = self.nete(X)
        return H

    def forward_er(self, X):
        """ Forward propagate through netR
        """
        H = self.nete(X)
        X_tilde = self.netr(H)
        return X_tilde

    def forward_g(self, Z):
        """ Forward propagate through netG
        """
        E_hat = self.netg(Z)
        return E_hat

    def forward_dg(self, E_Hat, H_Hat):
        """ Forward propagate through netD
        """
        Y_fake = self.netd(H_Hat)
        Y_fake_e = self.netd(E_Hat)
        return Y_fake, Y_fake_e

    def forward_rg(self, H_hat):
        """ Forward propagate through netG
        """
        X_Hat = self.netr(H_hat)
        return X_Hat

    def forward_s(self, H):
        """ Forward propagate through netS
        """
        H_supervise = self.nets(H)
        # print(self.H, self.H_supervise)
        return H_supervise

    def forward_sg(self, E_Hat):
        """ Forward propagate through netS
        """
        H_Hat = self.nets(E_Hat)
        return H_Hat

    def forward_d(self, H, H_Hat, E_Hat):
        """ Forward propagate through netD
        """
        Y_real = self.netd(H)
        Y_fake = self.netd(H_Hat)
        Y_fake_e = self.netd(E_Hat)
        return Y_real, Y_fake, Y_fake_e

    def backward_er(self, X_tilde, X):
        """ Backpropagate through netE
        """
        self.err_er = self.l_mse(X_tilde, X)
        self.err_er.backward(retain_graph=True)

    def backward_er_(self, X_tilde, X, H_supervise, H):
        """ Backpropagate through netE
        """
        self.err_er_ = self.l_mse(X_tilde, X)
        self.err_s = self.l_mse(H_supervise[:, :-1, :], H[:, 1:, :])
        self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
        self.err_er.backward(retain_graph=True)

    def backward_g(self, Y_fake, Y_fake_e, X_Hat, X, H, H_supervise):
        """ Backpropagate through netG
        """
        self.err_g_U = self.l_bce(Y_fake, torch.ones_like(Y_fake))

        self.err_g_U_e = self.l_bce(Y_fake_e, torch.ones_like(Y_fake_e))
        self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(X_Hat, [0])[1] + 1e-6) - torch.sqrt(
            torch.std(X, [0])[1] + 1e-6)))  # |a^2 - b^2|
        self.err_g_V2 = torch.mean(
            torch.abs((torch.mean(X_Hat, [0])[0]) - (torch.mean(X, [0])[0])))  # |a - b|
        self.err_s = self.l_mse(H_supervise[:, :-1, :], H[:, 1:, :])
        self.err_g = self.err_g_U + \
                     self.err_g_U_e * self._gamma + \
                     self.err_g_V1 * self._generator_loss_weight + \
                     self.err_g_V2 * self._generator_loss_weight + \
                     torch.sqrt(self.err_s)
        self.err_g.backward(retain_graph=True)

    def backward_s(self, H, H_supervise):
        """ Backpropagate through netS
        """
        self.err_s = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)

    def backward_d(self, Y_real, Y_fake, Y_fake_e):
        """ Backpropagate through netD
        """
        self.err_d_real = self.l_bce(Y_real, torch.ones_like(Y_real))
        self.err_d_fake = self.l_bce(Y_fake, torch.zeros_like(Y_fake))
        self.err_d_fake_e = self.l_bce(Y_fake_e, torch.zeros_like(Y_fake_e))
        self.err_d = self.err_d_real + \
                     self.err_d_fake + \
                     self.err_d_fake_e * self._gamma
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)

    def optimize_params_er(self, X):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        X_tilde = self.forward_er(X)

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er(X_tilde, X)
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self, X, H):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        X_tilde = self.forward_er(X)
        H_supervise = self.forward_s(H)
        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_(X_tilde, X, H_supervise, H)
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self, X):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        H = self.forward_e(X)
        H_supervise = self.forward_s(H)

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s(H, H_supervise)
        self.optimizer_s.step()

    def optimize_params_g(self, X, Z):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        H = self.forward_e(X)
        H_supervise = self.forward_s(H)
        E_Hat = self.forward_g(Z)
        H_Hat = self.forward_sg(E_Hat)
        X_Hat = self.forward_rg(H_Hat)
        Y_fake, Y_fake_e = self.forward_dg(E_Hat, H_Hat)

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g(Y_fake, Y_fake_e, X_Hat, X, H, H_supervise)
        self.optimizer_g.step()
        self.optimizer_s.step()

        return H

    def optimize_params_d(self, X, Z):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        H = self.forward_e(X)
        E_Hat = self.forward_g(Z)
        H_Hat = self.forward_sg(E_Hat)
        Y_real, Y_fake, Y_fake_e = self.forward_d(H, H_Hat, E_Hat)
        Y_fake, Y_fake_e = self.forward_dg(E_Hat, H_Hat)

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d(Y_real, Y_fake, Y_fake_e)
        self.optimizer_d.step()

    def _train(self, data):
        """ Train the model
        """

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Training'
            epoch_iterator.set_description(description.format(loss=0))

        train_dataloader = DataLoader(data, batch_size= self._batch_size, shuffle=True, drop_last= True)

        for epoch in epoch_iterator:
            for X, T in train_dataloader:
                #X = nn.utils.rnn.pack_padded_sequence(X, T, batch_first=True, enforce_sorted=False)
                X = X.to(self.device)

                # Train er
                self.nete.train()
                self.netr.train()

                # train encoder & decoder
                self.optimize_params_er(X)

                if self._wandb:
                    wandb.log({"Encoder & Decoder Loss": self.err_er})

            for X, T in train_dataloader:
                #X = nn.utils.rnn.pack_padded_sequence(X, T, batch_first=True, enforce_sorted=False)
                X = X.to(self.device)

                # Train s
                # self.nete.eval()
                self.nets.train()

                # train superviser
                self.optimize_params_s(X)

                if self._wandb:
                    wandb.log({"Supervisor Loss": self.err_s})

            for X, T in train_dataloader:
                #X = nn.utils.rnn.pack_padded_sequence(X, T, batch_first=True, enforce_sorted=False)
                X = X.to(self.device)
                T = T.to(self.device)
                Z = random_generator(self._batch_size, self._feature_noise_dim + self._attribute_noise_dim, T, self._max_sequence_len)
                #Z = nn.utils.rnn.pack_padded_sequence(Z, T, batch_first=True, enforce_sorted=False)
                Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

                # Train for one iter
                enc_dec_avg = []
                for kk in range(self._generator_steps):
                    self.netg.train()

                    # train superviser
                    H = self.optimize_params_g(X, Z)

                    # train er
                    self.nete.train()
                    self.netr.train()

                    # train encoder & decoder
                    self.optimize_params_er_(X, H)

                    enc_dec_avg.append(self.err_er_)

                self.netd.train()

                # train superviser
                self.optimize_params_d(X, Z)

                if self._wandb:
                    wandb.log({"Comb Supervisor Loss": self.err_d, "Comb Encoder & Decoder Loss": np.mean(enc_dec_avg)})

    def train_numpy(
            self,
            features: Union[np.ndarray, List[np.ndarray]],
            feature_sequence_length: np.ndarray,
            feature_types: Optional[List[OutputType]] = None,
            attributes: Optional[np.ndarray] = None,
            attribute_types: Optional[List[OutputType]] = None):
        
        if isinstance(features, np.ndarray):
            features = [seq for seq in features]

        if self._verbose:
            logging.info(
                f"features length={len(features)}, first sequence shape={features[0].shape}, dtype={features[0].dtype}",
                extra={"user_log": True},
            )
        if attributes is not None and self._verbose:
            logging.info(
                f"attributes shape={attributes.shape}, dtype={attributes.dtype}",
                extra={"user_log": True},
            )

        if attributes is not None:
            if attributes.shape[0] != len(features):
                raise RuntimeError(
                    "First dimension of attributes and features must be the same length, i.e., the number of training examples."  # noqa
                )

        if attributes is not None and attribute_types is None:
            # Automatically determine attribute types
            attribute_types = []
            for i in range(attributes.shape[1]):
                try:
                    # Here we treat integer columns as continuous, and thus the
                    # generated values will be (unrounded) floats. This may not
                    # be the right choice, and may be surprising to give integer
                    # inputs and get back floats. An explicit list of
                    # feature_types can be given (or constructed by passing
                    # discrete_columns to train_dataframe) to control this
                    # behavior. And we can look into a better fix in the future,
                    # maybe using # of distinct values, and having an explicit
                    # integer type so we appropriately round the final output.

                    # This snippet is only detecting types to construct
                    # feature_types, not making any changes to elements of
                    # features.
                    attributes[:, i].astype("float")
                    attribute_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    attribute_types.append(OutputType.DISCRETE)

        if feature_types is None:
            # Automatically determine feature types
            feature_types = []
            for i in range(features[0].shape[1]):
                try:
                    # Here we treat integer columns as continuous, see above
                    # comment.

                    # This snippet is only detecting types to construct
                    # feature_types, not making any changes to elements of
                    # features.
                    for seq in features:
                        seq[:, i].astype("float")
                    feature_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    feature_types.append(OutputType.DISCRETE)

        if not self.is_built:
            attribute_outputs, feature_outputs = create_outputs_from_data(
                attributes,
                features,
                attribute_types,
                feature_types,
                normalization=self._normalization,
                apply_feature_scaling=self._apply_feature_scaling,
                apply_example_scaling=self._apply_example_scaling,
            )
    
            self._build(
                attribute_outputs,
                feature_outputs,
            )

        continuous_features_ind = [
            ind
            for ind, val in enumerate(self.feature_outputs)
            if "ContinuousOutput" in str(val.__class__)
        ]

        if continuous_features_ind:
            # DGAN does not handle nans in continuous features (though in
            # categorical features, the encoding will treat nans as just another
            # category). To ensure we have none of these problematic nans, we
            # will interpolate to replace nans with actual float values, but if
            # we have too many nans in an example interpolation is unreliable.

            # Find valid examples based on minimal number of nans.
            valid_examples = validation_check(
                features,
                continuous_features_ind,
            )

            # Only use valid examples for the entire dataset.
            features = [seq for valid, seq in zip(valid_examples, features) if valid]
            features_seq_len = [seq for valid, seq in zip(valid_examples, feature_sequence_length) if valid]
            if attributes is not None:
                attributes = attributes[valid_examples]

            # Apply linear interpolations to replace nans for continuous
            # features:
            nan_linear_interpolation(features, continuous_features_ind)

        (
            internal_features,
            internal_additional_attributes,
        ) = transform_features(
            features, self.feature_outputs, self._max_sequence_len
        )

        if internal_additional_attributes is not None:
            if np.any(np.isnan(internal_additional_attributes)):
                raise RuntimeError(
                    f"NaN found in internal additional attributes."
                )
        else:
            internal_additional_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        if attributes is not None and self.attribute_outputs is not None:
            internal_attributes = transform_attributes(
                attributes,
                self.attribute_outputs,
            )
        else:
            internal_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        if self.attribute_outputs and np.any(np.isnan(internal_attributes)):
            raise RuntimeError(
                f"NaN found in internal attributes."
            )

        # As TimeGAN is not able to handle attributes merge them back into features
        internal_features_extdend = np.concatenate((internal_features,
                                                    np.repeat(np.expand_dims(internal_attributes, axis= 1), internal_features.shape[1], axis= 1),
                                                    np.repeat(np.expand_dims(internal_additional_attributes, axis= 1), internal_features.shape[1], axis= 1)), axis= 2)


        dataset = TensorDataset(
            torch.tensor(internal_features_extdend),
            torch.tensor(features_seq_len, dtype= torch.int),
        )

        self._train(dataset)
    
    def train_dataframe(self, df: pd.DataFrame,
                        attribute_columns: Optional[List[str]] = None,
                        feature_columns: Optional[List[str]] = None,
                        example_id_column: Optional[str] = None,
                        time_column: Optional[str] = None,
                        discrete_columns: Optional[List[str]] = None,
                        df_style: DfStyle = DfStyle.WIDE):
        
        if self.data_frame_converter is None:

            # attribute columns should be disjoint from feature columns
            if attribute_columns is not None and feature_columns is not None:
                if set(attribute_columns).intersection(set(feature_columns)):
                    raise ValueError(
                        "The `attribute_columns` and `feature_columns` lists must not have overlapping values!"
                    )

            if df_style == DfStyle.WIDE:
                self.data_frame_converter = _WideDataFrameConverter.create(
                    df,
                    attribute_columns=attribute_columns,
                    feature_columns=feature_columns,
                    discrete_columns=discrete_columns,
                )
            elif df_style == DfStyle.LONG:
                if time_column is not None and example_id_column is not None:
                    if time_column == example_id_column:
                        raise ValueError(
                            "The `time_column` and `example_id_column` values cannot be the same!"
                        )

                if example_id_column is not None:
                    # It should not be contained in any other lists
                    other_columns = set()
                    if discrete_columns is not None:
                        other_columns.update(discrete_columns)
                    if feature_columns is not None:
                        other_columns.update(feature_columns)
                    if attribute_columns is not None:
                        other_columns.update(attribute_columns)

                    if (example_id_column in other_columns or time_column in other_columns):
                        raise ValueError(
                            "The `example_id_column` and `time_column` must not be present in any other column lists!"
                        )

                # neither of these should be in any of the other lists
                if example_id_column is None and attribute_columns:
                    raise ValueError(
                        "Please provide an `example_id_column`, auto-splitting not available with only attribute columns."  # noqa
                    )
                if example_id_column is None and attribute_columns is None:
                    if self._verbose:
                        logging.warning(
                            f"The `example_id_column` was not provided, TIMEGAN will autosplit dataset into sequences of size {self._max_sequence_len}!"  # noqa
                        )
                    if len(df) < self._max_sequence_len:
                        raise ValueError(
                            f"Received {len(df)} rows in long data format, but TIMEGAN requires max_sequence_len={self._max_sequence_len} rows to make a training example. Note training will require at least 2 examples."  # noqa
                        )

                    df = df[
                        : math.floor(len(df) / self._max_sequence_len)
                        * self._max_sequence_len
                    ].copy()
                    if time_column is not None:
                        df[time_column] = pd.to_datetime(df[time_column])

                        df = df.sort_values(time_column)

                    example_id_column = "example_id"
                    df[example_id_column] = np.repeat(
                        range(len(df) // self._max_sequence_len),
                        self._max_sequence_len,
                    )

                self.data_frame_converter = _LongDataFrameConverter.create(
                    df,
                    max_sequence_len=self._max_sequence_len,
                    attribute_columns=attribute_columns,
                    feature_columns=feature_columns,
                    example_id_column=example_id_column,
                    time_column=time_column,
                    discrete_columns=discrete_columns,
                )
            else:
                raise ValueError(
                    f"df_style param must be an enum value DfStyle ('wide' or 'long'), received '{df_style}'"
                )

        attributes, features, features_seq_len = self.data_frame_converter.convert(df)

        self.train_numpy(
            attributes=attributes,
            features=features,
            feature_sequence_length= features_seq_len,
            attribute_types=self.data_frame_converter.attribute_types,
            feature_types=self.data_frame_converter.feature_types
        )

    def _generate(self, num_samples, sequence_lengths= None):
        if sequence_lengths:
            T = torch.tensor(sequence_lengths, dtype= torch.int).to(self.device)
        else:
            T = torch.full((num_samples, ), self._max_sequence_len, dtype= torch.int).to(self.device)

        if len(T) != num_samples:
            raise ValueError("Length of sequence_lengths must match num_samples")

        ## Synthetic data generation
        Z = random_generator(num_samples, self._feature_noise_dim + self._attribute_noise_dim, T, self._max_sequence_len)
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)
        E_hat = self.netg(Z)  # [?, 24, 24]
        H_hat = self.nets(E_hat)  # [?, 24, 24]
        generated_data = self.netr(H_hat).cpu().detach().numpy()  # [?, 24, 24]

        return generated_data, T.cpu().detach().numpy()

    def generate_dataframe(self,
        n: int):

        if not self.is_built:
            raise RuntimeError("Must build TimeGAN model prior to generating samples.")

        # Generate across multiple batches of batch_size. Use same size for
        # all batches and truncate the last partial batch at the very end
        # before returning.
        num_batches = n // self._batch_size
        if n % self._batch_size != 0:
            num_batches += 1

        batch_iterator = tqdm(range(num_batches), disable=(not self._verbose))
        if self._verbose:
            description = 'Generating'
            batch_iterator.set_description(description.format(loss=0))
        internal_data = []
        for batch in batch_iterator:
            temp_internal_data, temp_seq_len = self._generate(self._batch_size)
            internal_data.append(temp_internal_data)
        internal_data = np.concatenate(internal_data, axis=0)

        internal_feature_dim = sum([out.dim for out in self.feature_outputs])
        internal_features = internal_data[:, :, :internal_feature_dim]
        internal_attribute_dim = sum([out.dim for out in self.attribute_outputs])
        internal_attributes = internal_data[:, 0, internal_feature_dim:internal_feature_dim + internal_attribute_dim]
        internal_additional_attributes = internal_data[:, 0, internal_feature_dim + internal_attribute_dim:]

        attributes = None
        if internal_attributes is not None and self.attribute_outputs is not None:
            attributes = inverse_transform_attributes(
                internal_attributes,
                self.attribute_outputs,
            )

        if internal_features is None:
            raise RuntimeError(
                "Received None instead of internal features numpy array"
            )

        features = inverse_transform_features(
            internal_features,
            self.feature_outputs,
            additional_attributes = internal_additional_attributes
        )

        attributes = attributes[:n]
        features = features[:n]

        return self.data_frame_converter.invert(attributes, features)

class TIMEGANSynthesizer(LossValuesMixin, BaseSynthesizer):
    """Synthesizer for sequential data.

    This synthesizer uses the ``deepecho.models.par.PARModel`` class as the core model.
    Additionally, it uses a separate synthesizer to model and sample the context columns
    to be passed into PAR.

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
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        context_columns (list[str]):
            A list of strings, representing the columns that do not vary in a sequence.
        segment_size (int):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can be passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment.
        epochs (int):
            The number of epochs to train for. Defaults to 128.
        sample_size (int):
            The number of times to sample (before choosing and
            returning the sample which maximizes the likelihood).
            Defaults to 1.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'numerical': None,
        'boolean': None
    }

    def _get_context_metadata(self):
        context_columns_dict = {}
        context_columns = self.context_columns.copy() if self.context_columns else []
        if self._sequence_key:
            context_columns += self._sequence_key

        for column in context_columns:
            context_columns_dict[column] = self.metadata.columns[column]

        for column, column_metadata in self._extra_context_columns.items():
            context_columns_dict[column] = column_metadata

        context_metadata_dict = {'columns': context_columns_dict}
        return SingleTableMetadata.load_from_dict(context_metadata_dict)

    def __init__(self, metadata, max_sequence_len: int, enforce_min_max_values=True, enforce_rounding=False,
                 locales=['en_US'], context_columns=None, recursive_module= "gru", hidden_dim= 24, num_layers= 3,
                 metric_iteration=10, beta1= 0.9, learning_rate= 0.001, gamma= 1, encoder_loss_weight_s = 0.1,
                 normalization: Normalization = Normalization.ZERO_ONE, apply_feature_scaling: bool = True, apply_example_scaling: bool = True, binary_encoder_cutoff: int = 150,
                 encoder_loss_weight_0 = 10, generator_loss_weight = 100, generator_steps= 2,
                 epochs= 500000, batch_size= 16, device= "cuda", verbose= False, use_wandb= False):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        sequence_key = self.metadata.sequence_key
        self._sequence_key = list(_cast_to_iterable(sequence_key)) if sequence_key else None
        if not self._sequence_key:
            raise SynthesizerInputError(
                'TIMEGAN is designed for multi-sequence data, identifiable through a '
                'sequence key. Your metadata does not include a sequence key.'
            )

        sequenceKey_metadata_dict = metadata.to_dict()
        sequenceKey_metadata_dict['columns'] = {column: metadata.columns[column] for column in self._sequence_key}
        del sequenceKey_metadata_dict['sequence_index']
        sequenceKey_metadata = SingleTableMetadata.load_from_dict(sequenceKey_metadata_dict)
        self._sequenceKey_processor = DataProcessor(
            metadata=sequenceKey_metadata,
            enforce_rounding=False,
            enforce_min_max_values=False,
            locales=self.locales,
        )

        self._sequence_index = self.metadata.sequence_index
        self.context_columns = context_columns or []
        self._extra_context_columns = {}
        self.extended_columns = {}


        self._max_sequence_len = max_sequence_len

        self._model_kwargs = {
            'max_sequence_len': max_sequence_len,
            'recursive_module': recursive_module,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'metric_iteration': metric_iteration,
            'beta1': beta1,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'encoder_loss_weight_s': encoder_loss_weight_s,
            'normalization': normalization,
            'apply_feature_scaling': apply_feature_scaling,
            'apply_example_scaling': apply_example_scaling,
            'binary_encoder_cutoff': binary_encoder_cutoff,
            'encoder_loss_weight_0': encoder_loss_weight_0,
            'generator_loss_weight': generator_loss_weight,
            'generator_steps': generator_steps,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'wandb': use_wandb,
            'device': device
        }

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        for parameter_name, value in self._model_kwargs.items():
            instantiated_parameters[parameter_name] = value

        return instantiated_parameters

    def _validate_context_columns(self, data):
        errors = []
        if self.context_columns:
            for sequence_key_value, data_values in data.groupby(_groupby_list(self._sequence_key)):
                for context_column in self.context_columns:
                    if len(data_values[context_column].unique()) > 1:
                        errors.append((
                            f"Context column '{context_column}' is changing inside sequence "
                            f'({self._sequence_key}={sequence_key_value}).'
                        ))

        return errors

    def _validate(self, data):
        return self._validate_context_columns(data)

    def _transform_sequence_key(self, data):
        self._sequenceKey_processor.fit(data[self._sequence_key])

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidValueError:
                If a table of the data is not present in the metadata.
        """
        super().auto_assign_transformers(data)

        # Ensure that sequence index does not get auto assigned with enforce_min_max_values
        if self._sequence_index and self.get_transformers()[self._sequence_index]:
            sequence_index_transformer = self.get_transformers()[self._sequence_index]
            if sequence_index_transformer.enforce_min_max_values:
                sequence_index_transformer.enforce_min_max_values = False

    def _preprocess(self, data):
        """Transform the raw data to numerical space.

        For PAR, none of the sequence keys are transformed.

        Args:
            data (pandas.DataFrame):
                The raw data to be transformed.

        Returns:
            pandas.DataFrame:
                The preprocessed data.
        """

        self._extra_context_columns = {}
        sequence_key_transformers = {sequence_key: None for sequence_key in self._sequence_key}
        if not self._data_processor._prepared_for_fitting:
            self.auto_assign_transformers(data)

        self.update_transformers(sequence_key_transformers)
        preprocessed = super()._preprocess(data)

        if self._sequence_key:
            self._transform_sequence_key(preprocessed)

        return preprocessed

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.

        Raises:
            ValueError:
                Raise when the transformer of a context column is passed.
        """
        if set(column_name_to_transformer).intersection(set(self.context_columns)):
            raise SynthesizerInputError(
                'Transformers for context columns are not allowed to be updated.')

        super().update_transformers(column_name_to_transformer)



    def _fit(self, processed_data):
        """Fit this model to the data.

        Args:
            processed_data (pandas.DataFrame):
                pandas.DataFrame containing both the sequences,
                the entity columns and the context columns.
        """

        if len(self._sequence_key) > 1:
            new_sequence_key = str(uuid.uuid4().hex)
            processed_data[new_sequence_key] = processed_data[self._sequence_key].apply(
                lambda x: '_'.join(map(str, x)), axis=1)
            processed_data = processed_data.drop(columns=self._sequence_key)
            self._sequence_key = [new_sequence_key]

        self.data_columns = [
            column
            for column in processed_data.columns
            if column not in (
                    self._sequence_key + [self._sequence_index] + self.context_columns +
                    list(self._extra_context_columns.keys())
            )
        ]

        self._model = TimeGan(**self._model_kwargs)
        self._model.train_dataframe(attribute_columns= self.context_columns, feature_columns= self.data_columns, time_column= self._sequence_index, example_id_column= self._sequence_key[0], df= processed_data, df_style= DfStyle.LONG)

    def sample(self, num_rows, num_entities= None, conditions= None):
        """Sample new sequences.

        Args:
            num_sequences (int):
                Number of sequences to sample.
            sequence_length (int):
                If passed, sample sequences of this length. If ``None``, the sequence length will
                be sampled from the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same format as the fitted data.
        """

        if conditions is None:
            if not num_entities:
                num_entities = math.ceil(num_rows / self._max_sequence_len)
            if self._max_sequence_len * num_entities < num_rows:
                RuntimeError("Max sequence lenght * Num entities must be larger then num rows.")
            fake = self._model.generate_dataframe(num_entities)
            anonym_sequence_key = self._sequenceKey_processor._hyper_transformer.create_anonymized_columns(
                num_rows=num_entities,
                column_names= self._sequence_key
            )
            sequence_key_dict = dict(zip(fake[self._sequence_key[0]].unique(), anonym_sequence_key.transpose().values[0]))
            fake[self._sequence_key] = fake[self._sequence_key].replace(sequence_key_dict)
            return fake.sample(num_rows)

        raise NotImplementedError("DOPPELGANGERSynthesizer doesn't support conditional sampling.")
