"""DOPPELGANGER Synthesizer class."""
import math
import inspect
import logging
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import calendar

from .._utils import _cast_to_iterable, _groupby_list
from ..errors import SamplingError, SynthesizerInputError
from ..metadata.single_table import SingleTableMetadata
from ..sampling import Condition
from ..single_table.base import BaseSynthesizer
from ..single_table.ctgan import LossValuesMixin
from ..data_processing.data_processor import DataProcessor
from ..metadata.single_table import SingleTableMetadata
from ..single_table.utils import (log_numerical_distributions_error, validate_numerical_distributions)
from enum import Enum

import wandb

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import abc
from collections import Counter
from itertools import cycle
from scipy.stats import mode
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from category_encoders import BinaryEncoder, OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F


logger = logging.getLogger(__name__)

AttributeFeatureTriplet = Tuple[Optional[np.ndarray], List[np.ndarray], List[np.ndarray]]
NumpyArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]

## gretelai errors
class InternalError(RuntimeError):
    """
    Error that indicate invalid internal state.

    If you're using gretel_synthetics through documented interfaces, this usually
    indicates a bug in the gretel_synthetics itself.
    If you're using not documented interfaces, this could indicate invalid usage.

    This class of errors is equivalent to 5xx status codes in HTTP protocol.
    """

class DataError(ValueError):
    """
    Represents problems with training data before work is actually attempted.
    For example: data contains values that are not supported by the model that is
    being used: infinity, too many NaNs, nested data, etc.
    """

class ParameterError(ValueError):
    """
    Represents errors with configurations or parameter input to user-facing methods.
    For example: config referencing column that is not present in the data.
    """

class OutputType(Enum):
    """Supported variables types.

    Determines internal representation of variables and output layers in
    generation network.
    """

    DISCRETE = 0
    CONTINUOUS = 1

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

# gretlai transformations
"""Module for converting data to and from internal DGAN representation."""

def _new_uuid() -> str:
    """Return a random uuid prefixed with 'gretel-'."""
    return f"gretel-{uuid.uuid4().hex}"

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
            self._nan_proxy = _new_uuid()
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

def create_outputs_from_data(
    attributes: Optional[np.ndarray],
    features: List[np.ndarray],
    attribute_types: Optional[List[OutputType]],
    feature_types: Optional[List[OutputType]],
    normalization: Normalization,
    apply_feature_scaling: bool = False,
    apply_example_scaling: bool = False,
    binary_encoder_cutoff: int = 150,
) -> Tuple[Optional[List[Output]], List[Output]]:
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
) -> Output:
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
        raise ParameterError(
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
        logger.warning(
            f"The following discrete columns ({missing_discrete}) were not in the generated DataFrame, you may want to ensure this is intended!"  # noqa
        )

    return df

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
    def convert(self, df: pd.DataFrame) -> AttributeFeatureTriplet:
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

class _WideDataFrameConverter(_DataFrameConverter):
    """Convert "wide" format DataFrames.

    Expects one row for each example with 0 or more attribute columns and 1
    column per time point in the time series.
    """

    def __init__(
        self,
        attribute_columns: List[str],
        inp_feature_columns: List[str],
        tar_feature_columns: List[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        inp_feature_types: List[OutputType],
        tar_feature_types: List[OutputType],
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._inp_feature_columns = inp_feature_columns
        self._tar_feature_columns = tar_feature_columns
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._inp_feature_types = inp_feature_types
        self._tar_feature_types = tar_feature_types

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        attribute_columns: Optional[List[str]] = None,
        inp_feature_columns: Optional[List[str]] = None,
        tar_feature_columns: Optional[List[str]] = None,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Create a converter instance.

        See `train_dataframe` for parameter details.
        """
        if attribute_columns is None:
            attribute_columns = []
        else:
            attribute_columns = attribute_columns

        if inp_feature_columns is None:
            inp_feature_columns = [c for c in df.columns if c not in attribute_columns]
        else:
            inp_feature_columns = inp_feature_columns

        if tar_feature_columns is None:
            tar_feature_columns = [c for c in df.columns if c not in attribute_columns]
        else:
            tar_feature_columns = tar_feature_columns

        feature_columns = list(set(inp_feature_columns + tar_feature_columns))

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
        if all(c in discrete_column_set for c in inp_feature_columns):
            inp_feature_types = [OutputType.DISCRETE]
        else:
            inp_feature_types = [OutputType.CONTINUOUS]

        if all(c in discrete_column_set for c in tar_feature_columns):
            tar_feature_types = [OutputType.DISCRETE]
        else:
            tar_feature_types = [OutputType.CONTINUOUS]

        return _WideDataFrameConverter(
            attribute_columns=attribute_columns,
            inp_feature_columns=inp_feature_columns,
            tar_feature_columns=tar_feature_columns,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            inp_feature_types=inp_feature_types,
            tar_feature_types=tar_feature_types
        )

    @property
    def name(self) -> str:
        return "WideDataFrameConverter"

    @property
    def attribute_types(self):
        return self._attribute_types

    @property
    def feature_types(self):
        return self._inp_feature_types, self._tar_feature_types

    def convert(self, df: pd.DataFrame) -> AttributeFeatureTriplet:
        if self._attribute_columns:
            attributes = df[self._attribute_columns].to_numpy()
        else:
            attributes = None

        inp_features = np.expand_dims(df[self._inp_feature_columns].to_numpy(), axis=-1)
        tar_features = np.expand_dims(df[self._tar_feature_columns].to_numpy(), axis=-1)

        return attributes, [seq for seq in inp_features], [seq for seq in tar_features]

    def invert(
        self, attributes: Optional[np.ndarray], features: List[np.ndarray]
    ) -> pd.DataFrame:
        if self._attribute_columns:
            if attributes is None:
                raise InternalError(
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
            "inp_feature_columns": self._inp_feature_columns,
            "tar_feature_columns": self._tar_feature_columns,
            "discrete_columns": self._discrete_columns,
            "df_column_order": self._df_column_order,
            "attribute_types": self._attribute_types,
            "inp_feature_types": self._inp_feature_types,
            "tar_feature_types": self._tar_feature_types,
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
        inp_feature_columns: List[str],
        tar_feature_columns: List[str],
        example_id_column: Optional[str],
        time_column: Optional[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        inp_feature_types: List[OutputType],
        tar_feature_types: List[OutputType],
        time_column_values: Optional[List[str]],
        generation_flag_index: Optional[int] = None,
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._inp_feature_columns = inp_feature_columns
        self._tar_feature_columns = tar_feature_columns
        self._example_id_column = example_id_column
        self._time_column = time_column
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._inp_feature_types = inp_feature_types
        self._tar_feature_types = tar_feature_types
        self._time_column_values = time_column_values
        self._generation_flag_index = generation_flag_index

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        max_sequence_len: int,
        attribute_columns: Optional[List[str]] = None,
        inp_feature_columns: Optional[List[str]] = None,
        tar_feature_columns: Optional[List[str]] = None,
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

        if inp_feature_columns is None:
            # If not specified, use remaining columns in the data frame that
            # are not used elsewhere
            inp_feature_columns = [c for c in df.columns if c not in given_columns]
        else:
            inp_feature_columns = inp_feature_columns

        if tar_feature_columns is None:
            # If not specified, use remaining columns in the data frame that
            # are not used elsewhere
            tar_feature_columns = [c for c in df.columns if c not in given_columns]
        else:
            tar_feature_columns = tar_feature_columns

        # Add feature columns too, so given_columns contains all columns of df
        # that we are actually using
        feature_columns = list(set(inp_feature_columns + tar_feature_columns))
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
        inp_feature_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in inp_feature_columns
        ]
        tar_feature_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in tar_feature_columns
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
                    raise DataError(
                        f"Found sequence with length {item[1]}, longer than max_sequence_len={max_sequence_len}"
                    )
                elif item[1] < max_sequence_len:
                    has_variable_length_sequences = True

            if has_variable_length_sequences:
                generation_flag_index = len(inp_feature_types)
                inp_feature_types.append(OutputType.DISCRETE)

                generation_flag_index = len(tar_feature_types)
                tar_feature_types.append(OutputType.DISCRETE)

        return cls(
            attribute_columns=attribute_columns,
            inp_feature_columns= inp_feature_columns,
            tar_feature_columns= tar_feature_columns,
            example_id_column=example_id_column,
            time_column=time_column,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            inp_feature_types= inp_feature_types,
            tar_feature_types= tar_feature_types,
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
        return self._inp_feature_types, self._tar_feature_types

    def convert(self, df: pd.DataFrame) -> AttributeFeatureTriplet:

        if self._time_column is not None:
            sorted_df = df.sort_values(by=[self._time_column])
        else:
            sorted_df = df

        if self._example_id_column is not None:
            # Use example_id_column to split into separate time series
            df_inp_features = sorted_df[self._inp_feature_columns]
            df_tar_features = sorted_df[self._tar_feature_columns]

            inp_features = list(
                df_inp_features.groupby(sorted_df[self._example_id_column]).apply(
                    pd.DataFrame.to_numpy
                )
            )

            tar_features = list(
                df_tar_features.groupby(sorted_df[self._example_id_column]).apply(
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
                        raise DataError(
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
            inp_features = [sorted_df[self._inp_feature_columns].to_numpy()]
            tar_features = [sorted_df[self._tar_feature_columns].to_numpy()]

            # Check that attributes are the same for all rows (since they are
            # all implicitly in the same example)
            for column in self._attribute_columns:
                if sorted_df[column].nunique() != 1:
                    raise DataError(f"Attribute {column} is not constant for all rows.")

            if self._attribute_columns:
                # With one example, attributes should all be constant, so grab from
                # the first row. Need to add first (example) dimension.
                attributes = np.expand_dims(
                    sorted_df[self._attribute_columns].iloc[0, :].to_numpy(), axis=0
                )
            else:
                attributes = None

        if self._generation_flag_index is not None:
            inp_features = [
                _add_generation_flag(seq, self._generation_flag_index)
                for seq in inp_features
            ]
            tar_features = [
                _add_generation_flag(seq, self._generation_flag_index)
                for seq in tar_features
            ]
        return attributes, inp_features, tar_features

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
                raise InternalError(
                    "Unable to invert features back to data frame, "
                    + f"converter expected {len(self._feature_columns)} features, "
                    + f"received numpy array with {seq.shape[1]}"
                )

            seq_column_parts = [seq]
            if self._attribute_columns:
                if attributes is None:
                    raise InternalError(
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
                    raise InternalError(
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
            "inp_feature_columns": self._inp_feature_columns,
            "tar_feature_columns": self._tar_feature_columns,
            "example_id_column": self._example_id_column,
            "time_column": self._time_column,
            "df_column_order": self._df_column_order,
            "discrete_columns": self._discrete_columns,
            "attribute_types": self._attribute_types,
            "inp_feature_types": self._inp_feature_types,
            "tar_feature_types": self._tar_feature_types,
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
        raise DataError(
            f"More than {100*invalid_examples_ratio_cutoff}% invalid examples in the continuous features. Please reduce the ratio of the NaNs and try again!"  # noqa
        )

    if (~valid_examples).any():
        logger.warning(
            f"There are {sum(~valid_examples)} examples that have too many nan values in numeric features, accounting for {np.mean(~valid_examples)*100}% of all examples. These invalid examples will be omitted from training.",  # noqa
            extra={"user_log": True},
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

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.float32)
    return mask  # (seq_len, seq_len)

def create_padding_mask(seq):
    seq = (seq.sum(dim=2) == 0).float()
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)

def create_masks(tar):
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return combined_mask, dec_target_padding_mask

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.size()[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(dff, activation='relu'),
        nn.Linear(d_model)
    )

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out1)

        return out3, attn_weights_block1


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, inp_dim, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.input_layer = nn.Sequential(
            nn.Linear(inp_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask, padding_mask):
        x = self.input_layer(x)

        seq_len = x.size(1)
        attention_weights = {}

        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1

        return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, inp_dim_dict: dict, net_dim_dict: dict, feature_order: list, net_activations, model_dim= 128, num_layers_enc = None, num_layers_dec= 4, num_heads= 2, dff= 128,
                 maximum_position_encoding= 256, net_info= None, final_dim= None, rate=0.1, learning_rate=0.001, verbose= False, epochs= 10, batch_size= 64, early_stop= 2):

        self._epochs = epochs
        self._batch_size = batch_size
        self._early_stop = early_stop

        self._learning_rate = learning_rate
        self._verbose = verbose

        self.feature_order = feature_order

        self.inp_dim_dict = inp_dim_dict
        self.inp_dim = len(inp_dim_dict.values())

        self.net_dim_dict = net_dim_dict
        self.net_dim = len(net_dim_dict.values())

        self.net_activations = net_activations

        if set(inp_dim_dict.keys()) == set(feature_order):
            raise InternalError(
                f"all elements in feature order need to be present in the input dict"
            )

        if set(inp_dim_dict.keys()) == set(net_dim_dict.keys()):
            raise InternalError(
                f"input dimension dict and input network dict need to have the same keys"
            )

        if self.inp_dim != len(self.net_activations):
            raise InternalError(
                f"An activation function needs to be provided for every input dimension"
            )

        self.decoder = Decoder(num_layers_dec, model_dim, num_heads, dff, maximum_position_encoding, self.inp_dim, rate)

        self.final_layer = nn.Linear(model_dim, model_dim)

        self.input_layer = {}
        prev_inp_dim = self.model_dim
        for name, dim in self.net_dim_dict.items():
            if name in self.net_activations.keys():
                if self.net_activations[name] is "relu":
                   self.input_layer[name] = nn.Sequential(
                        nn.Linear(prev_inp_dim, dim),
                        nn.ReLU()
                    )
                elif self.net_activations[name] is "tanh":
                    self.input_layer[name] = nn.Sequential(
                        nn.Linear(prev_inp_dim, dim),
                        nn.Tanh()
                    )
                elif self.net_activations[name] is "sigmoid":
                    self.input_layer[name] = nn.Sequential(
                        nn.Linear(prev_inp_dim, dim),
                        nn.Sigmoid()
                    )
                else:
                    self.input_layer[name] = nn.Linear(prev_inp_dim, dim)
            else:
                self.input_layer[name] = nn.Linear(prev_inp_dim, dim)
            prev_inp_dim += dim

        self.train_loss = nn.MSELoss()
        self.results = {x: [] for x in ["loss", "val_loss", "val_loss_full"]}

    def forward(self, input, training, look_ahead_mask, dec_padding_mask):
        tar_inp = input[:, :-1]
        tar_out = input[:, 1:]

        dec_output, attention_weights = self.decoder(tar_inp, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        preds = {}

        prev_dim = 0
        for net_name in self.feature_order:
            pred = self.input_layer[net_name](final_output)
            preds[net_name] = pred

            start = prev_dim
            end = prev_dim + self.inp_dim_dict[net_name]
            prev_dim += self.inp_dim_dict[net_name]
            to_add = tar_out[:, :, start:end]

            final_output = torch.cat([final_output, to_add], dim=-1)

        return preds, attention_weights

    def train_step(self, inp, tar, optimizer):
        combined_mask, dec_padding_mask = create_masks(tar)

        optimizer.zero_grad()
        predictions, _ = self.forward(inp, True, combined_mask, dec_padding_mask)
        loss = self.loss_function(tar, predictions)

        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self, inp, tar):
        combined_mask, dec_padding_mask = create_masks(tar)

        predictions, _ = self(inp, False, combined_mask, dec_padding_mask)
        loss = self.loss_function(tar, predictions)

        loss.backward()

        return loss.item()

class BANKSFORMER:
    def __init__(self, max_sequence_len: int, sample_len: int, dropout_rate= 0.1, d_model= 128, num_layers_dec = 4, num_heads = 2,
                 normalization = False, apply_feature_scaling = True, apply_example_scaling = False,
                 batch_size: int = 64, epochs: int = 80, early_stop: int = 2, cuda: bool = True, verbose= False
    ):
        # Preprocessing
        self._normalization= normalization
        self._apply_feature_scaling= apply_feature_scaling
        self._apply_example_scaling= apply_example_scaling

        # Model structure
        self._max_sequence_len= max_sequence_len
        self._sample_len= sample_len

        self._dropout_rate= dropout_rate
        self._d_model= d_model
        self._num_layers_dec= num_layers_dec
        self._num_heads= num_heads

        # Training parameters
        self._batch_size= batch_size
        self._epochs= epochs
        self._early_stop= early_stop

        self._cuda= cuda
        self._verbose= verbose

        self.is_built = False

        self.data_frame_converter = None

    def train_numpy(
        self,
        features_inp: Union[np.ndarray, List[np.ndarray]],
        features_tar: Union[np.ndarray, List[np.ndarray]],
        feature_inp_types: Optional[List[OutputType]] = None,
        feature_tar_types: Optional[List[OutputType]] = None,
        attributes: Optional[np.ndarray] = None,
        attribute_types: Optional[List[OutputType]] = None
    ) -> None:

        if isinstance(features_inp, np.ndarray):
            features_inp = [seq for seq in features_inp]

        if isinstance(features_tar, np.ndarray):
            features_tar = [seq for seq in features_tar]

        if len(features_inp) != len(features_tar):
            raise InternalError(
                "First dimension of two feature vectors must be the same length, i.e., the number of training examples."
                # noqa
            )

        if attributes is not None:
            if attributes.shape[0] != len(features_inp) or attributes.shape[0] != len(features_tar):
                raise InternalError(
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

        if feature_inp_types is None:
            # Automatically determine feature types
            features_inp_types = []
            for i in range(features_inp[0].shape[1]):
                try:
                    # Here we treat integer columns as continuous, see above
                    # comment.

                    # This snippet is only detecting types to construct
                    # feature_types, not making any changes to elements of
                    # features.
                    for seq in features_inp:
                        seq[:, i].astype("float")
                    feature_inp_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    feature_inp_types.append(OutputType.DISCRETE)

        if feature_tar_types is None:
            # Automatically determine feature types
            features_tar_types = []
            for i in range(features_tar[0].shape[1]):
                try:
                    # Here we treat integer columns as continuous, see above
                    # comment.

                    # This snippet is only detecting types to construct
                    # feature_types, not making any changes to elements of
                    # features.
                    for seq in features_tar:
                        seq[:, i].astype("float")
                    feature_tar_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    feature_tar_types.append(OutputType.DISCRETE)


        if not self.is_built:
            if self._verbose:
                logger.info(
                    "Determining outputs metadata from input data", extra={"user_log": True})

            attribute_outputs, feature_inp_outputs = create_outputs_from_data(
                attributes,
                features_inp,
                attribute_types,
                feature_inp_types,
                normalization=self._normalization,
                apply_feature_scaling=self._apply_feature_scaling,
                apply_example_scaling=self._apply_example_scaling,
            )
            _, feature_tar_outputs = create_outputs_from_data(
                None,
                features_tar,
                attribute_types,
                feature_tar_types,
                normalization= self._normalization,
                apply_feature_scaling=self._apply_feature_scaling,
                apply_example_scaling=self._apply_example_scaling,
            )



            logger.info("Building Banksformer networks", extra={"user_log": True})
            self._build(
                attribute_outputs,
                feature_inp_outputs,
                feature_tar_outputs
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
            logger.info(
                f"Checking for nans in the {len(continuous_features_ind)} numeric columns",
                extra={"user_log": True},
            )

            # Find valid examples based on minimal number of nans.
            valid_examples = validation_check(
                features,
                continuous_features_ind,
            )

            # Only use valid examples for the entire dataset.
            features = [seq for valid, seq in zip(valid_examples, features) if valid]
            if attributes is not None:
                attributes = attributes[valid_examples]

            logger.info(
                "Applying linear interpolations for nans (does not mean nans are present)",
                extra={"user_log": True},
            )
            # Apply linear interpolations to replace nans for continuous
            # features:
            nan_linear_interpolation(features, continuous_features_ind)

        logger.info("Creating encoded array of features", extra={"user_log": True})
        (
            internal_features,
            internal_additional_attributes,
        ) = transform_features(
            features, self.feature_outputs, self._max_sequence_len
        )

        if internal_additional_attributes is not None:
            if np.any(np.isnan(internal_additional_attributes)):
                raise InternalError(
                    f"NaN found in internal additional attributes."
                )
        else:
            internal_additional_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        logger.info("Creating encoded array of attributes", extra={"user_log": True})
        if attributes is not None and self.attribute_outputs is not None:
            internal_attributes = transform_attributes(
                attributes,
                self.attribute_outputs,
            )
        else:
            internal_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        logger.info(
            f"internal_features shape={internal_features.shape}, dtype={internal_features.dtype}",
            extra={"user_log": True},
        )
        logger.info(
            f"internal_additional_attributes shape={internal_additional_attributes.shape}, dtype={internal_additional_attributes.dtype}",
            extra={"user_log": True},
        )
        logger.info(
            f"internal_attributes shape={internal_attributes.shape}, dtype={internal_attributes.dtype}",
            extra={"user_log": True},
        )

        if self.attribute_outputs and np.any(np.isnan(internal_attributes)):
            raise InternalError(
                f"NaN found in internal attributes."
            )

        logger.info("Creating TensorDataset", extra={"user_log": True})
        dataset = TensorDataset(
            torch.Tensor(internal_attributes),
            torch.Tensor(internal_additional_attributes),
            torch.Tensor(internal_features),
        )

        logger.info("Calling _train()", extra={"user_log": True})
        self._train(dataset, verbose= self._verbose)

    def train_dataframe(
        self,
        df: pd.DataFrame,
        feature_inp_columns: Optional[List[str]] = None,
        feature_tar_columns: Optional[List[str]] = None,
        attribute_columns: Optional[List[str]] = None,
        example_id_column: Optional[str] = None,
        time_column: Optional[str] = None,
        discrete_columns: Optional[List[str]] = None,
        df_style: DfStyle = DfStyle.WIDE
    ) -> None:
        """Train DGAN model on data in pandas DataFrame.

        Training data can be in either "wide" or "long" format. "Wide" format
        uses one row for each example with 0 or more attribute columns and 1
        column per time point in the time series. "Wide" format is restricted to
        1 feature variable. "Long" format uses one row per time point, supports
        multiple feature variables, and uses additional example id to split into
        examples and time column to sort.

        Args:
            df: DataFrame of training data
            attribute_columns: list of column names containing attributes, if None,
                no attribute columns are used. Must be disjoint from
                the feature columns.
            feature_inp_columns: list of column names containing inp features, if None
                all non-attribute columns are used. Must be disjoint from
                attribute columns.
            feature_tar_columns: list of column names containing tar features, if None
                all non-attribute columns are used. Must be disjoint from
                attribute columns.
            example_id_column: column name used to split "long" format data
                frame into multiple examples, if None, data is treated as a
                single example. This value must be unique from the other
                column list parameters.
            time_column: column name used to sort "long" format data frame,
                if None, data frame order of rows/time points is used. This
                value must be unique from the other column list parameters.
            discrete_columns: column names (either attributes or features) to
                treat as discrete (use one-hot or binary encoding), any string
                or object columns are automatically treated as discrete
            df_style: str enum of "wide" or "long" indicating format of the
                DataFrame
        """

        feature_columns = list(set(feature_inp_columns + feature_tar_columns))

        if self.data_frame_converter is None:

            # attribute columns should be disjoint from feature columns
            if attribute_columns is not None and feature_inp_columns is not None and feature_tar_columns is not None:
                if set(attribute_columns).intersection(set(feature_inp_columns)) or set(attribute_columns).intersection(set(feature_tar_columns)):
                    raise ParameterError(
                        "The `attribute_columns` and `feature_columns` lists must not have overlapping values!"
                    )

            if df_style == DfStyle.WIDE:
                self.data_frame_converter = _WideDataFrameConverter.create(
                    df,
                    attribute_columns=attribute_columns,
                    inp_feature_columns= feature_inp_columns,
                    tar_feature_columns= feature_tar_columns,
                    discrete_columns=discrete_columns,
                )
            elif df_style == DfStyle.LONG:
                if time_column is not None and example_id_column is not None:
                    if time_column == example_id_column:
                        raise ParameterError(
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
                        raise ParameterError(
                            "The `example_id_column` and `time_column` must not be present in any other column lists!"
                        )

                # neither of these should be in any of the other lists
                if example_id_column is None and attribute_columns:
                    raise ParameterError(
                        "Please provide an `example_id_column`, auto-splitting not available with only attribute columns."  # noqa
                    )
                if example_id_column is None and attribute_columns is None:
                    if self._verbose:
                        logging.warning(
                            f"The `example_id_column` was not provided, DGAN will autosplit dataset into sequences of size {self._max_sequence_len}!"  # noqa
                        )
                    if len(df) < self._max_sequence_len:
                        raise DataError(
                            f"Received {len(df)} rows in long data format, but DGAN requires max_sequence_len={self._max_sequence_len} rows to make a training example. Note training will require at least 2 examples."  # noqa
                        )

                    df = df[: math.floor(len(df) / self._max_sequence_len)* self._max_sequence_len].copy()
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
                    inp_feature_columns= feature_inp_columns,
                    tar_feature_columns= feature_tar_columns,
                    example_id_column=example_id_column,
                    time_column=time_column,
                    discrete_columns=discrete_columns,
                )
            else:
                raise ParameterError(
                    f"df_style param must be an enum value DfStyle ('wide' or 'long'), received '{df_style}'"
                )

        logger.info(
            "Converting from DataFrame to numpy arrays", extra={"user_log": True}
        )
        attributes, features_inp, features_tar  = self.data_frame_converter.convert(df)

        logger.info("Calling train_numpy()", extra={"user_log": True})
        feature_inp_types, feature_tar_types = self.data_frame_converter.feature_types
        self.train_numpy(
            attributes=attributes,
            features_inp= features_inp,
            features_tar= features_tar,
            attribute_types=self.data_frame_converter.attribute_types,
            feature_inp_types= feature_inp_types,
            feature_tar_types= feature_tar_types
        )

    def generate_numpy(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> AttributeFeatureTriplet:
        """Generate synthetic data from DGAN model.

        Once trained, a DGAN model can generate arbitrary amounts of
        synthetic data by sampling from the noise distributions. Specify either
        the number of records to generate, or the specific noise vectors to use.

        Args:
            n: number of examples to generate
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            Tuple of attributes and features as numpy arrays.
        """

        if not self.is_built:
            raise InternalError("Must build DGAN model prior to generating samples.")

        if n is not None:
            # Generate across multiple batches of batch_size. Use same size for
            # all batches and truncate the last partial batch at the very end
            # before returning.
            num_batches = n // self._batch_size
            if n % self._batch_size != 0:
                num_batches += 1

            internal_data_list = []
            for _ in range(num_batches):
                internal_data_list.append(
                    self._generate(
                        self.attribute_noise_func(self._batch_size),
                        self.feature_noise_func(self._batch_size),
                    )
                )
            # Convert from list of tuples to tuple of lists with zip(*) and
            # concatenate into single numpy arrays for attributes, additional
            # attributes (if present), and features.
            #
            # NOTE: Despite linter complaints, the np.array(d) == None statement
            # is special because it checks for None values along the array like so:
            # In [4]: np.array([1,2,3,4]) == None
            # Out[4]: array([False, False, False, False])
            internal_data = tuple(
                (
                    np.concatenate(d, axis=0)
                    if not (np.array(d) == None).any()  # noqa
                    else None
                )
                for d in zip(*internal_data_list)
            )

        else:
            if attribute_noise is None or feature_noise is None:
                raise InternalError(
                    "generate() must receive either n or both attribute_noise and feature_noise"
                )
            attribute_noise = attribute_noise.to(self.device, non_blocking=True)
            feature_noise = feature_noise.to(self.device, non_blocking=True)

            internal_data = self._generate(attribute_noise, feature_noise)

        (
            internal_attributes,
            internal_additional_attributes,
            internal_features,
        ) = internal_data

        attributes = None
        if internal_attributes is not None and self.attribute_outputs is not None:
            attributes = inverse_transform_attributes(
                internal_attributes,
                self.attribute_outputs,
            )

        if internal_features is None:
            raise InternalError(
                "Received None instead of internal features numpy array"
            )

        features = inverse_transform_features(
            internal_features,
            self.feature_outputs,
            additional_attributes=internal_additional_attributes,
        )
        # Convert to list of numpy arrays to match primary input to train_numpy
        features = [seq for seq in features]

        if n is not None:
            if attributes is None:
                features = features[:n]
                return None, features
            else:
                return attributes[:n], features[:n]

        return attributes, features

    def generate_dataframe(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data from DGAN model.

        Once trained, a DGAN model can generate arbitrary amounts of
        synthetic data by sampling from the noise distributions. Specify either
        the number of records to generate, or the specific noise vectors to use.

        Args:
            n: number of examples to generate
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            pandas DataFrame in same format used in 'train_dataframe' call
        """

        attributes, features = self.generate_numpy(n, attribute_noise, feature_noise)

        return self.data_frame_converter.invert(attributes, features)

    def _build(
        self,
        attribute_outputs: Optional[List[Output]],
        feature_inp_outputs: List[Output],
        feature_tar_outputs: List[Output]
    ):
        """Setup internal structure for DGAN model.

        Args:
            attribute_outputs: custom metadata for attributes
            feature_outputs: custom metadata for features
        """

        self.attribute_outputs = attribute_outputs
        self.feature_inp_outputs = feature_inp_outputs
        self.feature_tar_outputs = feature_tar_outputs

        if self._cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.transformer = Transformer(
            self.feature_inp_outputs,
            self.feature_tar_outputs,
            activations=None,
            num_layers_enc=None,
            num_layers_dec=4,
            d_model=128,
            num_heads=2,
            dff=128,
            maximum_position_encoding=256,
            net_info=None,
            final_dim=None,
            rate=0.1
        )

        self.transformer.to(self.device)

        if self.attribute_outputs is None:
            self.attribute_outputs = []
        attribute_dim = sum(output.dim for output in self.attribute_outputs)

        self.is_built = True

    def _train(
        self,
        dataset: Dataset,
        verbose: bool = False,
    ):
        """Internal method for training DGAN model.

        Expects data to already be transformed into the internal representation
        and wrapped in a torch Dataset. The torch Dataset consists of 3-element
        tuples (attributes, additional_attributes, features). If attributes and/or
        additional_attribtues were not passed by the user, these indexes of the
        tuple will consists of nan-filled tensors which will later be filtered
        out and ignored in the DGAN training process.

        Args:
            dataset: torch Dataset containing tuple of (attributes, additional_attributes, features)
        """
        if len(dataset) <= 1:
            raise DataError(
                f"DGAN requires multiple examples to train, received {len(dataset)} example."
                + "Consider splitting a single long sequence into many subsequences to obtain "
                + "multiple examples for training."
            )

        # Our optimization setup does not work on batches of size 1. So if
        # drop_last=False would produce a last batch of size of 1, we use
        # drop_last=True instead.
        drop_last = len(dataset) % self._batch_size == 1

        loader = DataLoader(
            dataset,
            self._batch_size,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.transformer._learning_rate)

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Loss. ({loss:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        for epoch in epoch_iterator:

            for batch_no, (inp, tar) in enumerate(loader):

                combined_mask, dec_padding_mask = create_masks(tar)

                optimizer.zero_grad()

                predictions, _ = self.transformer(inp, True, combined_mask, dec_padding_mask)
                loss = self.transformer.loss_function(tar, predictions)

                loss.backward()
                optimizer.step()

                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(loss=loss.item())
                    )
                if self._use_wandb:
                    wandb.log({"Loss": loss.item()})


    def set_random_state(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)

class BANKSFORMERSynthesizer(LossValuesMixin, BaseSynthesizer):
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
        context_columns (List[str]):
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

    def __init__(self, metadata, max_sequence_len: int, sample_len: int, enforce_min_max_values=True, enforce_rounding=False,
                    locales=['en_US'], context_columns=None, amount_column= None, attribute_noise_dim: int = 10, feature_noise_dim: int = 10, attribute_num_layers: int = 3,
                    attribute_num_units: int = 100, feature_num_layers: int = 1, feature_num_units: int = 100, use_attribute_discriminator: bool = True, apply_feature_scaling: bool = True, apply_example_scaling: bool = True,
                    binary_encoder_cutoff: int = 150, forget_bias: bool = False, gradient_penalty_coef: float = 10.0, attribute_gradient_penalty_coef: float = 10.0,
                    attribute_loss_coef: float = 1.0, generator_learning_rate: float = 0.001, generator_beta1: float = 0.5, discriminator_learning_rate: float = 0.001,
                    discriminator_beta1: float = 0.5, attribute_discriminator_learning_rate: float = 0.001, attribute_discriminator_beta1: float = 0.5,
                    batch_size: int = 1024, epochs: int = 400, discriminator_rounds: int = 1, generator_rounds: int = 1, cuda: bool = True, mixed_precision_training: bool = False, verbose= False,
                    numerical_distributions=None, default_distribution= "norm"):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales
        )

        sequence_key = self.metadata.sequence_key
        self._sequence_key = list(_cast_to_iterable(sequence_key)) if sequence_key else None
        if not self._sequence_key:
            raise SynthesizerInputError(
                'BANKSFORMER is designed for multi-sequence data, identifiable through a '
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
        self._amount_column = amount_column

        self._inp_columns = []
        self._tar_columns = []

        self._max_sequence_len = max_sequence_len

        validate_numerical_distributions(numerical_distributions, self.metadata.columns)

        self.numerical_scaler = {}
        self.categorical_encoder = {}

        self._model_kwargs = {
            'max_sequence_len' : max_sequence_len,
            'sample_len': sample_len,
            'batch_size' : batch_size,
            'epochs' : epochs,
            'cuda' : cuda,
            'verbose' : verbose
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

    def _bulk_encode_time_value(self, val, max_val):
        x = np.sin(2 * np.pi / max_val * val)
        y = np.cos(2 * np.pi / max_val * val)
        return np.stack([x, y], axis=1)
    def _transform_sequence_index(self, data):
        # convert date to individual components
        data[self._sequence_index] = pd.to_datetime(data[self._sequence_index])
        data["td"] = data[[self._sequence_key[0], self._sequence_index]].groupby(self._sequence_key).diff()
        data["td"] = data["td"].apply(lambda x: x.days)
        data["td"] = data["td"].fillna(0.0)

        data["month"] = data[self._sequence_index].dt.month
        data = pd.concat((data, pd.DataFrame(self._bulk_encode_time_value(data["month"].values, 12), columns= ["month-i1", "month-i2"])), axis= 1)
        data["day"] = data[self._sequence_index].dt.day
        data = pd.concat((data, pd.DataFrame(self._bulk_encode_time_value(data["day"].values, 31), columns= ["day-i1", "day-i2"])), axis= 1)
        data["dow"] = data[self._sequence_index].dt.dayofweek
        data = pd.concat((data, pd.DataFrame(self._bulk_encode_time_value(data["dow"].values, 7), columns= ["dow-i1", "dow-i2"])), axis= 1)
        data["dtme"] = data[self._sequence_index].apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)
        data = pd.concat((data, pd.DataFrame(self._bulk_encode_time_value(data["dtme"].values, 31), columns= ["dtme-i1", "dtme-i2"])), axis= 1)

        self._inp_columns += ["td", "month-i1", "month-i2", "day-i1", "day-i2", "dow-i1", "dow-i2", "dtme-i1", "dtme-i2"]
        self._tar_columns += ["td", "month", "day", "dow", "dtme"]

        return data

    def _reverse_transform_sequence_index(self, data):
        # convert individual components to date

        return data

    def _transform_sequence_key(self, data):
        self._sequenceKey_processor.fit(data[self._sequence_key])

    def _tranform_amount_column(self, data):
        data[self._amount_column] = np.log10(data[self._amount_column] + 1)
        return data

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        super().auto_assign_transformers(data)

        # Ensure that sequence index does not get auto assigned with enforce_min_max_values
        if self._sequence_index:
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
        if len(self._sequence_key) > 1:
            new_sequence_key = str(uuid.uuid4().hex)
            data[new_sequence_key] = data[self._sequence_key].apply(lambda x: '_'.join(map(str, x)), axis=1)
            data = data.drop(columns=self._sequence_key)
            self._sequence_key = [new_sequence_key]

        self._extra_context_columns = {}
        sequence_key_transformers = {sequence_key: None for sequence_key in self._sequence_key}
        if not self._data_processor._prepared_for_fitting:
            self.auto_assign_transformers(data)

        self.update_transformers(sequence_key_transformers)

        if self._amount_column:
            data = self._tranform_amount_column(data)

        preprocessed = super()._preprocess(data)

        for column in preprocessed.columns:
            if self.metadata.columns[column]["sdtype"] == "numerical" and column not in self._sequence_key and column not in list(self._extra_context_columns.keys()) and column != self._sequence_index:
                self.numerical_scaler[column] = StandardScaler()
                preprocessed[column] = self.numerical_scaler[column].fit_transform(preprocessed[[column]]).reshape(-1)
                self._inp_columns.append(column)
                self._tar_columns.append(column)

            elif self.metadata.columns[column]["sdtype"] == "categorical" and column not in self.context_columns and column not in self._sequence_key and column not in list(self._extra_context_columns.keys()) and column != self._sequence_index:
                enc = OneHotEncoder()
                enc.fit(preprocessed[[column]])
                preprocessed = pd.concat((preprocessed, enc.transform(preprocessed[[column]])), axis= 1)
                self.categorical_encoder[column] = enc
                self._inp_columns += enc.feature_names_out_

            elif self.metadata.columns[column]["sdtype"] == "categorical":
                enc = LabelEncoder()
                enc.fit(preprocessed[[column]])
                preprocessed[column] = enc.transform(preprocessed[[column]])
                self.categorical_encoder[column] = enc
                if column not in self.context_columns and column not in list(self._extra_context_columns.keys()):
                    self._tar_columns.append(column)

            elif column not in self.context_columns and column not in self._sequence_key and column not in list(self._extra_context_columns.keys()) and column != self._sequence_index:
                self._inp_columns.append(column)
                self._tar_columns.append(column)

        if self._sequence_index and self.metadata.to_dict()["columns"][self._sequence_index]["sdtype"] == "datetime":
            preprocessed = self._transform_sequence_index(preprocessed)

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

        self._model = BANKSFORMER(**self._model_kwargs)
        self._model.train_dataframe(attribute_columns= self.context_columns, feature_inp_columns= self._inp_columns, feature_tar_columns= self._tar_columns, time_column= self._sequence_index, example_id_column= self._sequence_key[0], df= processed_data, df_style= DfStyle.LONG)

    def sample(self, num_rows, conditions= None):
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
            num_sequences = math.ceil(num_rows / self._max_sequence_len)
            fake = self._model.generate_dataframe(num_sequences)

            if self._sequence_index and self.metadata.to_dict()["columns"][self._sequence_index]["sdtype"] == "datetime":
                ##TODO: change date back to normal
                pass

            anonym_sequence_key = self._sequenceKey_processor._hyper_transformer.create_anonymized_columns(
                num_rows=num_sequences,
                column_names=self._sequence_key
            )
            sequence_key_dict = dict(zip(fake[self._sequence_key[0]].unique(), anonym_sequence_key.transpose().values[0]))
            fake[self._sequence_key] = fake[self._sequence_key].replace(sequence_key_dict)

            return fake[:num_rows]

        raise NotImplementedError("CTGANSynthesizer doesn't support conditional sampling.")
