import cupy as cp
import numpy as np
import pandas as pd

class Linkage():
  """
  A class for linking records between two Pandas DataFrames based on previously estimated conditional match probabilities.
  
  :param df_A: The first DataFrame containing records to be linked.
  :type df_A: pandas.DataFrame
  :param df_B: The second DataFrame containing records to be linked.
  :type df_B: pandas.DataFrame
  :param Indices: List of arrays, where each array contains the indices of record pairs from ``df_A`` and ``df_B`` corresponding to a specific pattern of discrete similarity levels across variables.
  :type Indices: list[cupy.ndarray]
  :param Ksi: Array of conditional match probabilities for all patterns of discrete similarity levels across variables.
  :type Ksi: numpy.ndarray
  """

  def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, Indices, Ksi: np.array):

    self.df_A = df_A
    self.df_B = df_B
    self.Indices = Indices
    self.Ksi = Ksi

  def transform(self, Threshold = 0.85):
    """
    Returns a DataFrame containing all pairs of records from ``df_A`` and ``df_B`` whose conditional match probabilities exceed a specified threshold.

    :param Threshold: Threshold value above which pairs of records from ``df_A`` and ``df_B`` are considered matches. Defaults to 0.85.
    :type Threshold: float, optional
    :return: A DataFrame linking all pairs of records from ``df_A`` and ``df_B`` with conditional match probabilities greater than the specified threshold.
    :rtype: pandas.DataFrame
    :raises Exception: If no pairs of records have conditional match probabilities exceeding the threshold.
    """

    mempool = cp.get_default_memory_pool()

    # Adding suffixes and indices to df_A and df_B
    df_A = self.df_A.add_suffix("_A")

    df_B = self.df_B.add_suffix("_B")

    df_A["Index_A"] = range(len(df_A))

    df_B["Index_B"] = range(len(df_B))

    # Extracting the Indices for which Ksi is above the threshold
    Patterns_Above_Threshold = np.ravel(np.argwhere(self.Ksi >= Threshold))
    
    if np.sum([self.Indices[i - 1].size for i in Patterns_Above_Threshold]) == 0:
      raise Exception("No pair of observations has a conditional match probability exceeding the threshold.")
    
    Indices_to_Link = cp.concatenate((self.Indices[i - 1] for i in Patterns_Above_Threshold))

    Indices_to_Link_A = Indices_to_Link // len(df_B)

    Indices_to_Link_A_cpu = Indices_to_Link_A.get()

    Indices_to_Link_B = Indices_to_Link % len(df_B)

    Indices_to_Link_B_cpu = Indices_to_Link_B.get()

    del Indices_to_Link, Indices_to_Link_A, Indices_to_Link_B
    mempool.free_all_blocks()

    # Extracting the records in df_A with which records in df_B must be linked
    df_A = df_A.iloc[Indices_to_Link_A_cpu,:]

    df_A["Index_B"] = Indices_to_Link_B_cpu

    return df_A.merge(df_B, on = "Index_B")
