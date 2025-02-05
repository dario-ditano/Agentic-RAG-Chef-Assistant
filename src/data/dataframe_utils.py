import pandas as pd


def keep_first_n_rows(n: int, complete_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the first n rows of the provided DataFrame and returns it as a new DataFrame.
    This function is helpful when working with large datasets to extract a subset
    based on the given row count.

    :param n: Number of rows to keep from the beginning of the DataFrame.
    :param complete_df: The complete DataFrame from which a subset is selected.
    :type complete_df: pandas.DataFrame
    :return: A new DataFrame containing the first n rows of the original
        DataFrame.
    :rtype: pandas.DataFrame
    """
    return complete_df.head(n)
