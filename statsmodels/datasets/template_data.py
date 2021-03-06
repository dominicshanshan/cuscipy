"""Name of dataset."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """E.g., This is public domain."""
TITLE       = """Title of the dataset"""
SOURCE      = """
This section should provide a link to the original dataset if possible and
attribution and correspondance information for the dataset's original author
if so desired.
"""

DESCRSHORT  = """A short description."""

DESCRLONG   = """A longer description of the dataset."""

#suggested notes
NOTE        = """
::

    Number of observations:
    Number of variables:
    Variable name definitions:

Any other useful information that does not fit into the above categories.
"""


def load(as_pandas=None):
    """
    Load the data and return a Dataset class instance.

    Parameters
    ----------
    as_pandas : bool
        Flag indicating whether to return pandas DataFrames and Series
        or cupy recarrays and arrays.  If True, returns pandas.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return du.as_cupy_dataset(load_pandas(), as_pandas=as_pandas)


def load_pandas():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def _get_data():
    return du.load_csv(__file__, 'DatasetName.csv')
