from functools import partial, update_wrapper, wraps

import pandas as pd
import logging

# from full.features import FEATURE_FUNCS

LUNAR_NEW_YEAR = {
    2000: {"month": 2, "day": 5},
    2001: {"month": 1, "day": 24},
    2002: {"month": 2, "day": 12},
    2003: {"month": 2, "day": 1},
    2004: {"month": 1, "day": 22},
    2005: {"month": 2, "day": 9},
    2006: {"month": 1, "day": 29},
    2007: {"month": 2, "day": 18},
    2008: {"month": 2, "day": 7},
    2009: {"month": 1, "day": 26},
    2010: {"month": 2, "day": 14},
    2011: {"month": 2, "day": 3},
    2012: {"month": 1, "day": 23},
    2013: {"month": 2, "day": 10},
    2014: {"month": 1, "day": 31},
    2015: {"month": 2, "day": 19},
    2016: {"month": 2, "day": 8},
    2017: {"month": 1, "day": 28},
    2018: {"month": 2, "day": 16},
    2019: {"month": 2, "day": 5},
    2020: {"month": 1, "day": 25},
    2021: {"month": 2, "day": 12},
    2022: {"month": 2, "day": 1},
    2023: {"month": 1, "day": 22},
    2024: {"month": 2, "day": 10},
    2025: {"month": 1, "day": 29},
    2026: {"month": 2, "day": 17},
    2027: {"month": 2, "day": 6},
    2028: {"month": 1, "day": 26},
    2029: {"month": 2, "day": 13},
    2030: {"month": 2, "day": 3},
    2031: {"month": 1, "day": 23},
    2032: {"month": 2, "day": 11},
    2033: {"month": 1, "day": 31},
    2034: {"month": 2, "day": 19},
    2035: {"month": 2, "day": 8},
    2036: {"month": 1, "day": 28},
    2037: {"month": 2, "day": 15},
    2038: {"month": 2, "day": 4},
    2039: {"month": 1, "day": 24},
    2040: {"month": 2, "day": 12},
    2041: {"month": 2, "day": 1},
    2042: {"month": 1, "day": 22},
    2043: {"month": 2, "day": 10},
    2044: {"month": 1, "day": 30},
    2045: {"month": 2, "day": 17},
    2046: {"month": 2, "day": 6},
    2047: {"month": 1, "day": 26},
    2048: {"month": 2, "day": 14},
    2049: {"month": 2, "day": 2},
    2050: {"month": 1, "day": 23},
    2051: {"month": 2, "day": 11},
    2052: {"month": 2, "day": 1},
    2053: {"month": 2, "day": 19},
    2054: {"month": 2, "day": 8},
    2055: {"month": 1, "day": 28},
    2056: {"month": 2, "day": 15},
    2057: {"month": 2, "day": 4},
    2058: {"month": 1, "day": 24},
    2059: {"month": 2, "day": 12},
    2060: {"month": 2, "day": 2},
    2061: {"month": 1, "day": 21},
    2062: {"month": 2, "day": 9},
    2063: {"month": 1, "day": 29},
    2064: {"month": 2, "day": 17},
    2065: {"month": 2, "day": 5},
    2066: {"month": 1, "day": 26},
    2067: {"month": 2, "day": 14},
    2068: {"month": 2, "day": 3},
    2069: {"month": 1, "day": 23},
    2070: {"month": 2, "day": 11},
    2071: {"month": 1, "day": 31},
    2072: {"month": 2, "day": 19},
    2073: {"month": 2, "day": 7},
    2074: {"month": 1, "day": 27},
    2075: {"month": 2, "day": 15},
    2076: {"month": 2, "day": 5},
    2077: {"month": 1, "day": 24},
    2078: {"month": 2, "day": 12},
    2079: {"month": 2, "day": 2},
    2080: {"month": 1, "day": 22},
    2081: {"month": 2, "day": 9},
    2082: {"month": 1, "day": 29},
    2083: {"month": 2, "day": 17},
    2084: {"month": 2, "day": 6},
    2085: {"month": 1, "day": 26},
    2086: {"month": 2, "day": 14},
    2087: {"month": 2, "day": 3},
    2088: {"month": 1, "day": 24},
    2089: {"month": 2, "day": 10},
    2090: {"month": 1, "day": 30},
    2091: {"month": 2, "day": 18},
    2092: {"month": 2, "day": 7},
    2093: {"month": 1, "day": 27},
    2094: {"month": 2, "day": 15},
    2095: {"month": 2, "day": 5},
    2096: {"month": 1, "day": 25},
    2097: {"month": 2, "day": 12},
    2098: {"month": 2, "day": 1},
    2099: {"month": 1, "day": 21},
}


def timing_log():
    """
    Wrapper to logger statements keeping track of wrapped
    function's execution time.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # wrapper.deps = deps

        return wrapper

    return decorator


def add_deps_attr(deps=None):
    """
    Wrapper to add a 'deps' attribute to the
    wrapped function. `deps` stands for 'dependencies',
    and it is a list of other feature names that the feature
    depends on.

    This can be used as a building block to build dependency
    graphs for proper composition of feature functions.

    Parameters
    ----------
    deps: List[str], optional
        List of feature columns that the decorated feature
        depends on.

    Returns
    -------
    wrapped function with 'deps' attribute.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.deps = deps

        return wrapper

    return decorator


def add_cols_attr(cols=None):
    """
    Wrapper to add a "cols" attribute to the wrapped function.
    If

    Parameters
    ----------
    cols : List[str], optional
        List of columns that the feature function generates
        on an input dataframe. If `None`, it will use
        the function's `__name__` attribute as the sole
        entry of the `cols` list.

    Returns
    -------
    wrapped function with 'cols' attribute.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if cols is None:
            wrapper.cols = [wrapper.__name__]
        else:
            wrapper.cols = cols
        return wrapper

    return decorator


def wrapped_partial(func, *args, **kwargs):
    """
    Enhances functools.partial with a call to
    functools.update_wrapper so the returned
    partial object has the same attributes as
    the original function that was partially
    initialized.
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def get_entity(data, entity_id, copy=True):
    """
    Retrieves time series from data using the entity_id
    dictionary as identifiers.

    Parameters
    ----------
    data : pandas.DataFrame

    entity_id : dict
        Dictionary identifying a modeling entity. The keys and values
        of the dictionary are expected to correspond to columns and
        their values in a digital-combo-like table.

    copy: bool, default True
        Whether or not to return a copy of the dataframe.

    Returns
    -------
    pandas.DataFrame
        Subset of input dataframe.
    """

    # ====== select data subset for this target cut ======
    ts = data[(pd.Series(entity_id) == data[list(entity_id)]).all(axis=1)]

    if copy:
        return ts.copy()

    return ts


def ordered_dedup(seq):
    """
    Deduplicates a sequence while maintaining order.

    Notes
    -----
    `set(seq)` does not guarantee original ordering.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def featurize(
    source_table, feature_functions, record_key, output_report=False
):
    """
    Featurize source table on given feature functions.

    Parameters
    ----------
    source_table : pd.DataFrame
        Table of raw data required for featurization
    feature_names : list
        Names of feature functions to apply. These names are
        used against a project-owned lookup to retrieve
        actual feature functions.
    record_key : List[str]
        Record key for incoming source table to be used for
        feature functions that require grouping.
    output_report : Boolean, default False
        In addition to the feature table, return a list of
        the successful/failed functions and and columns.

    Returns
    -------
    pd.DataFrame
        Feature table consisting of feature columns
        calculated on top of source table.
    successful_functions: list
        list of functions that were successfully computed
    failed_functions: list
        list of functions that were unsuccessful.
    successful_columns : list
        list of columns that were successfully generated
    """

    # 2) iterate over unique feature functions
    logging.debug("Featurizing source data.")
    feature_frames = []
    failed_functions = []
    successful_functions = []
    successful_columns = []
    for feature_func in feature_functions:
        logging.debug(f"Applying feature function {feature_func}.")

        # apply feature function
        try:
            feature_frame = feature_func(source_table)

        except Exception as e:
            logging.error(
                f"Failed to apply feature function {feature_func} - {e}"
            )
            failed_functions.append(feature_func)
            continue

        # extract feature columns, reset index and append to frame
        # TODO: Enforce feature_func.cols to align with FEATURE_COL_TO_FUNC
        try:
            feature_cols = feature_func.cols
            feature_frame = feature_frame[record_key + feature_cols].set_index(
                record_key
            )
            # feature_frame = feature_frame.loc[
            #     ~feature_frame.index.duplicated(keep="first")
            # ]
            feature_frames.append(feature_frame)
        except Exception as e:
            logging.error(
                f"Failed to retrieve feature columns for feature function {feature_func} - {e}."
            )
            continue

        # log successes
        successful_functions.append(feature_func)
        successful_columns.extend(feature_cols)

    # 3) concatenate dataframes into feature table
    try:
        feature_data = pd.concat(feature_frames, axis=1).reset_index(
            drop=False
        )
    except Exception as e:
        logging.error(f"Failed to concatenate feature frames - {e}")
        raise

    # 4) restrict feature table to successful columns
    try:
        feature_data = feature_data[record_key + successful_columns]
    except Exception as e:
        logging.error(
            f"Failed to restrict feature data to feature column set: {successful_columns} - {e}"
        )
        raise

    # optional return
    if output_report:
        return (
            feature_data,
            successful_functions,
            failed_functions,
            successful_columns,
        )

    else:
        return feature_data
