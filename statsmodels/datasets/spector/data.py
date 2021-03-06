"""Spector and Mazzeo (1980) - Program Effectiveness Data"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission of the original author, who
retains all rights. """
TITLE       = __doc__
SOURCE      = """
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm

The raw data was downloaded from Bill Greene's Econometric Analysis web site,
though permission was obtained from the original researcher, Dr. Lee Spector,
Professor of Economics, Ball State University."""

DESCRSHORT  = """Experimental data on the effectiveness of the personalized
system of instruction (PSI) program"""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of Observations - 32

    Number of Variables - 4

    Variable name definitions::

        Grade - binary variable indicating whether or not a student's grade
                improved.  1 indicates an improvement.
        TUCE  - Test score on economics test
        PSI   - participation in program
        GPA   - Student's grade point average
"""


def load(as_pandas=None):
    """
    Load the Spector dataset and returns a Dataset class instance.

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
    Load the Spector dataset and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=3)


def _get_data():
    data = du.load_csv(__file__, 'spector.csv', sep=' ')
    data = du.strip_column_names(data)
    data = data.iloc[:, [1, 2, 3, 4]]
    return data.astype(float)
