import cupy as cp
import numpy as np
import pandas as pd

class Linkage():
  '''
  This class links the records in two data frames based on previously estimated conditional match probabilities.
  '''

  def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, Indices, Ksi: np.array):
    '''
    Arguments:
    ----------
    - df_A (Pandas DataFrame): The first dataframe.
    - df_B (Pandas DataFrame): The second dataframe.
    - Indices (list of CuPy arrays): This list contains the indices of pairs of records in df_A and df_B corresponding to each pattern of discrete levels of similarity across variables.
    - Ksi (NumPy array): This array contains the conditional match probability for each pattern of discrete levels of similarity across variables.
    '''
    self.df_A = df_A
    self.df_B = df_B
    self.Indices = Indices
    self.Ksi = Ksi

  def transform(self, Threshold = 0.5):
    '''
    This method returns a dataframe in which all pairs of observations with conditional match probabilities above some threshold are linked.

    Arguments:
    ----------
    - Threshold (float): This is the threshold above which pairs of observations in df_A and df_B must be linked. Its value must be between 0 and 1.

    Returns:
    --------
    - A dataframe in which all pairs of records in df_A and df_B with a conditional match probability above the threshold are linked.
    '''

    mempool = cp.get_default_memory_pool()

    # Adding suffixes and indices to df_A and df_B

    df_A = self.df_A.add_suffix('_A')

    df_B = self.df_B.add_suffix('_B')

    df_A['_Index_A'] = range(len(df_A))

    df_B['_Index_B'] = range(len(df_B))

    # Extracting the Indices for which Ksi is above the threshold

    Indices_to_Link = cp.concatenate((self.Indices[i] for i in range(len(self.Indices)) if self.Ksi[i] >= Threshold))

    Indices_to_Link_A = Indices_to_Link % len(df_B)

    Indices_to_Link_A_cpu = Indices_to_Link_A.get()

    Indices_to_Link_B = Indices_to_Link // len(df_B)

    Indices_to_Link_B_cpu = Indices_to_Link_B.get()

    del Indices_to_Link, Indices_to_Link_A, Indices_to_Link_B
    mempool.free_all_blocks()

    # Extracting the records in df_A with which records in df_B must be linked

    df_A = df_A.iloc[Indices_to_Link_A_cpu,:]

    # Adding the index of the record in df_B to which each record in df_A should be linked

    df_A['_Index_B'] = Indices_to_Link_B_cpu

    # Adding the conditional match probability associated with each pair of records

    df_A['_Ksi'] = np.repeat([self.Ksi[i] for i in range(len(self.Indices)) if self.Ksi[i] >= Threshold], [len(x) for x in [self.Indices[i] for i in range(len(self.Indices)) if self.Ksi[i] >= Threshold]])

    # Linking the records in df_A and df_B

    # Move columns around: _Index_A and _Index_B and _Ksi first, A, and then B

    return df_A.merge(df_B, on = '_Index_B').sort_values(by = ['_Index_A', '_Index_B'])
