"""Synthesizers for Single Table data."""

from .copulagan import CopulaGANSynthesizer
from .copulas import GaussianCopulaSynthesizer
from .ctgan import CTGANSynthesizer
from .tvae import TVAESynthesizer
from .tvae_parallel import TVAESynthesizer as TVAESynthesizer_parallel
from .wgangp import WGANGPSynthesizer, WGANGP_DRSSynthesizer
from .findiff import FINDIFFSynthesizer
from .gmm import GMMSynthesizer

__all__ = (
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'TVAESynthesizer_parallel',
    'CopulaGANSynthesizer',
    'WGANGPSynthesizer',
    'WGANGP_DRSSynthesizer',
    'FINDIFFSynthesizer',
    'GMMSynthesizer'
)
