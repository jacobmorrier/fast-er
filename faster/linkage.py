import cupy as cp
import numpy as np
import pandas as pd

class Linkage():
  """
  This class links the records in two data frames based on previously estimated conditional match probabilities.
  """

  def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, Indices, Ksi: np.array):
    """
    :param df_A: The first dataframe.
    :type df_A: pd.DataFrame
    :param df_B: The second dataframe.
    :type df_B: pd.DataFrame
    :param Indices: List containing the indices of pairs of records in df_A and df_B corresponding to each pattern of discrete levels of similarity across variables.
    :type Indices: list of cp.array
    :param Ksi: Array containing the conditional match probabilities for all patterns of discrete levels of similarity across variables.
    :type Ksi: np.array
    """

    self.df_A = df_A
    self.df_B = df_B
    self.Indices = Indices
    self.Ksi = Ksi

  def transform(self, Threshold = 0.85):
    """
    This method returns a dataframe in which all pairs of observations with conditional match probabilities above some threshold are linked.

    :param Threshold: Threshold above which pairs of observations in df_A and df_B must be linked, defaults to 0.85.
    :type Threshold: float, optional
    :return: Dataframe in which all pairs of records in df_A and df_B with a conditional match probability above the threshold are linked.
    """

    mempool = cp.get_default_memory_pool()

    # Adding suffixes and indices to df_A and df_B
    df_A = self.df_A.add_suffix('_A')

    df_B = self.df_B.add_suffix('_B')

    df_A['Index_A'] = range(len(df_A))

    df_B['Index_B'] = range(len(df_B))

    # Extracting the Indices for which Ksi is above the threshold
    Indices_to_Link = cp.concatenate((self.Indices[i - 1] for i in np.argwhere(self.Ksi >= Threshold).T[0]))

    Indices_to_Link_A = Indices_to_Link % len(df_B)

    Indices_to_Link_A_cpu = Indices_to_Link_A.get()

    Indices_to_Link_B = Indices_to_Link // len(df_B)

    Indices_to_Link_B_cpu = Indices_to_Link_B.get()

    del Indices_to_Link, Indices_to_Link_A, Indices_to_Link_B
    mempool.free_all_blocks()

    # Extracting the records in df_A with which records in df_B must be linked
    df_A = df_A.iloc[Indices_to_Link_A_cpu,:].reset_index()

    df_A['Index_B'] = Indices_to_Link_B_cpu

    return df_A.merge(df_B, on = 'Index_B').reset_index()
