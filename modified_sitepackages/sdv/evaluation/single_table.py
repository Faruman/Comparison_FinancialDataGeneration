"""Methods to compare the real and synthetic data for single-table."""
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from sdmetrics import visualization
from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.single_table.quality_report import QualityReport
from sdmetrics.reports.base_report import BaseReport

from sdv.errors import VisualizationUnavailableError

from sklearn.preprocessing import LabelEncoder

class KolmogorovSmirnovTest():
    """
    Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    def __init__(self, **kwargs) -> None:
        pass
    @staticmethod
    def name() -> str:
        return "ks_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame, metadata: dict):
        res = []
        for col in X_gt.columns:
            statistic, _ = ks_2samp(X_gt[col], X_syn[col])
            res.append(1 - statistic)

        return float(np.mean(res))

class JensenShannonDistance():
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""
    def __init__(self, normalize: bool = True, **kwargs) -> None:
        self.normalize = normalize

    @staticmethod
    def name() -> str:
        return "jensenshannon_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _evaluate_stats(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame, metadata: dict):
        stats_gt = {}
        stats_syn = {}
        stats_ = {}

        for col, val in metadata["columns"].items():
            if val["sdtype"] == "categorical":
                stats_gt[col], stats_syn[col] = X_gt.value_counts(dropna=False, normalize=self.normalize
                ).align(
                    X_syn.value_counts(dropna=False, normalize=self.normalize),
                    join="outer",
                    axis=0,
                    fill_value=0,
                )
            else:
                local_bins = min(100, len(X_gt[col].unique()))
                X_gt_bin, gt_bins = pd.cut(X_gt[col], bins=local_bins, retbins=True)
                X_syn_bin = pd.cut(X_syn[col], bins=gt_bins)
                stats_gt[col], stats_syn[col] = X_gt_bin.value_counts(
                    dropna=False, normalize=self.normalize
                ).align(
                    X_syn_bin.value_counts(dropna=False, normalize=self.normalize),
                    join="outer",
                    axis=0,
                    fill_value=0,
                )
                stats_gt[col] += 1
                stats_syn[col] += 1

            stats_[col] = jensenshannon(stats_gt[col], stats_syn[col])
            if np.isnan(stats_[col]):
                raise RuntimeError("NaNs in prediction")

        return stats_, stats_gt, stats_syn

    def evaluate(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
        metadata: dict
    ):
        stats_, _, _ = self._evaluate_stats(X_gt, X_syn, metadata= metadata)

        return sum(stats_.values()) / len(stats_.keys())

class SimilarityReport(BaseReport):
    """Single table quality report.

    This class creates a similarity report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        super().__init__()

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        if not isinstance(metadata, dict):
            raise TypeError('The provided metadata is not a dictionary.')

        self._validate(real_data, synthetic_data, metadata)

        columns_to_compare = [key for key, value in metadata['columns'].items() if value["sdtype"] not in ["id"]]
        metadata['columns'] = {key: value for key, value in metadata['columns'].items() if key in columns_to_compare}

        real_data = real_data[columns_to_compare]
        synthetic_data = synthetic_data[columns_to_compare]

        for key, value in metadata['columns'].items():
            if value["sdtype"] == "categorical":
                le = LabelEncoder()
                le.fit(real_data[key])
                real_data[key] = le.transform(real_data[key])
                synthetic_data[key] = le.transform(synthetic_data[key])

        jsd = JensenShannonDistance()
        jsd_score = jsd.evaluate(real_data, synthetic_data, metadata= metadata)

        kst = KolmogorovSmirnovTest()
        kst_score = kst.evaluate(real_data, synthetic_data, metadata= metadata)

        self._properties = {"Property": ["Jensen Shannon Distance", "Kolmogorov Smirnov Test"],
                            "Score": [jsd_score, kst_score]}
        self.is_generated = True
    def get_properties(self):
        self._check_report_generated()
        return pd.DataFrame(self._properties)

def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Single table quality report object.
    """
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report

def evaluate_similarity(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Single table quality report object.
    """
    similariy_report = SimilarityReport()
    similariy_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return similariy_report

def run_diagnostic(real_data, synthetic_data, metadata, verbose=True):
    """Run diagnostic report for the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        DiagnosticReport:
            Single table diagnostic report object.
    """
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report


def get_column_plot(real_data, synthetic_data, metadata, column_name, plot_type=None):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.DataFrame):
            The synthetic table data.
        metadata (SingleTableMetadata):
            The table metadata.
        column_name (str):
            The name of the column.
        plot_type (str or None):
            The plot to be used. Can choose between ``distplot``, ``bar`` or ``None``. If ``None`
            select between ``distplot`` or ``bar`` depending on the data that the column contains,
            ``distplot`` for datetime and numerical values and ``bar`` for categorical.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    sdtype = metadata.columns.get(column_name)['sdtype']
    if plot_type is None:
        if sdtype in ['datetime', 'numerical']:
            plot_type = 'distplot'
        elif sdtype in ['categorical', 'boolean']:
            plot_type = 'bar'

        else:
            raise VisualizationUnavailableError(
                f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                'supported visualization. To visualize this data anyways, please add a '
                "'plot_type'."
            )

    if sdtype == 'datetime':
        datetime_format = metadata.columns.get(column_name).get('datetime_format')
        real_data = pd.DataFrame({
            column_name: pd.to_datetime(real_data[column_name], format=datetime_format)
        })
        synthetic_data = pd.DataFrame({
            column_name: pd.to_datetime(synthetic_data[column_name], format=datetime_format)
        })

    return visualization.get_column_plot(
        real_data,
        synthetic_data,
        column_name,
        plot_type=plot_type
    )


def get_column_pair_plot(
        real_data, synthetic_data, metadata, column_names, plot_type=None, sample_size=None):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        metadata (SingleTableMetadata):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter`` or ``None``.
            If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending on the data
            that the column contains, ``scatter`` used for datetime and numerical values,
            ``heatmap`` for categorical and ``box`` for a mix of both. Defaults to ``None``.
        sample_size (int or None):
            The number of samples to use for the plot. If ``None`` use the whole dataset.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    real_data = real_data.copy()
    synthetic_data = synthetic_data.copy()
    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            sdtype = metadata.columns.get(column_name)['sdtype']
            if sdtype in ['numerical', 'datetime']:
                plot_type.append('scatter')
            elif sdtype in ['categorical', 'boolean']:
                plot_type.append('heatmap')
            else:
                raise VisualizationUnavailableError(
                    f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                    'supported visualization. To visualize this data anyways, please add a '
                    "'plot_type'."
                )

        if len(set(plot_type)) > 1:
            plot_type = 'box'
        else:
            plot_type = plot_type.pop()

    for column_name in column_names:
        sdtype = metadata.columns.get(column_name)['sdtype']
        if sdtype == 'datetime':
            datetime_format = metadata.columns.get(column_name).get('datetime_format')
            real_data[column_name] = pd.to_datetime(
                real_data[column_name],
                format=datetime_format
            )
            synthetic_data[column_name] = pd.to_datetime(
                synthetic_data[column_name],
                format=datetime_format
            )

    require_subsample = sample_size and sample_size < min(len(real_data), len(synthetic_data))
    if require_subsample:
        real_data = real_data.sample(n=sample_size)
        synthetic_data = synthetic_data.sample(n=sample_size)

    return visualization.get_column_pair_plot(
        real_data,
        synthetic_data,
        column_names,
        plot_type
    )
