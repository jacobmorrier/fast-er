import cupy as cp
import functools
import math
import numpy as np
import pandas as pd

jaro_winkler_code = r"""
extern "C"{

__device__ float jaro_winkler(const char *str1,
                              const int len1,
                              bool *hash_str1,
                              const char *str2,
                              const int len2,
                              bool *hash_str2) {

    // Description: Computes the Jaro-Winkler distance between two strings

    // Inputs:
    // - str1: First string
    // - len1: Length of str1
    // - hash_str1: Working memory to keep track of which characters in str1 are
    //              matching to corresponding characters in str2
    // - str2: Second string
    // - len2: Length of str2
    // - hash_str2: Working memory to keep track of which characters in str2 are
    //              matching to corresponding characters in str1

    // Output:
    // - dist: Jaro-Winkler distance between str1 and str2

    // If either string is null, the Jaro-Winkler distance between str1 and str2 is 0

    if (len1 == 0 || len2 == 0) {

        return 0.0;

    } else {

        // We compute the number of matching characters between str1 and str2

        // We consider the characters max(len1, len2) / 2 - 1 away from each other

        int max_dist = max(len1, len2) / 2 - 1;

        int match = 0;

        for (int i = 0; i < len1; i++) {

            for (int j = max(0, i - max_dist); j < min(len2, i + max_dist + 1); j++) {

                if (str1[i] == str2[j] && hash_str2[j] == false) {

                    // Two characters are matching if they appear in both strings
                    // at most max_dist characters away from each other

                    hash_str1[i] = true;
                    hash_str2[j] = true;
                    match++;
                    break;

                }

            }

        }

        // If there is no matching characters between both strings,
        // the Jaro-Winkler distance between them is 0

        if (match == 0) {

            return 0.0;

        } else {

            // If a positive number of matching characters is found, we need to
            // compute the number of transpositions, that is, the number of matching
            // characters that are not in the right order divided by two

            float t = 0;

            int point = 0;

            for (int i = 0; i < len1; i++) {

                if (hash_str1[i] == true) {

                    while (hash_str2[point] == false) {

                        point++;

                    }

                    if (str1[i] != str2[point]) {

                        t++;

                    }

                    point++;

                }

            }

            t /= 2;

            // The Jaro distance between str1 and str2 is defined as follows:

            float dist;

            dist = (((float)match / (float)len1) + ((float)match / (float)len2) + (((float)match - t) / (float)match)) / 3.0;

            // To go from the Jaro distance to the Jaro-Winkler distance, we need
            // to compute the length of the common prefix between both strings

            int prefix = 0;

            for (int i = 0; i < min(min(len1, len2), 4); i++) {

                if (str1[i] == str2[i]) {

                    prefix++;

                } else {

                    break;

                }

            }

            // To obtain the Jaro-Winkler distance, we adjust the Jaro distance
            // for the length of the common prefix between both strings

            dist += 0.1 * prefix * (1 - dist);

            return dist;

        }

    }

}

__global__ void jaro_winkler_kernel(char *str1,
                                    int *offsets1,
                                    bool *buffer1,
                                    int n1,
                                    char *str2,
                                    int *offsets2,
                                    bool *buffer2,
                                    int n2,
                                    float *output) {

    // Inputs:
    // - str1: First vector of strings stored as an arrow (i.e., concatenated
    //         next to each other)
    // - offsets1: Vector storing the index where each string in str1 starts
    // - buffer1: Working memory to keep track of which characters in str1 are
    //            matching to corresponding characters in str2
    // - n1: Number of strings contained in str1
    // - str2: Second vector of strings stored as an arrow
    // - offsets2: Vector storing the index where each string in str2 starts
    // - buffer2: Working memory to keep track of which characters in str2 are
    //            matching to corresponding characters in str1
    // - n2: Number of strings contained in str2
    // - output: Vector storing the computed Jaro-Winkler distances

    const int id = threadIdx.x + blockDim.x * blockIdx.x;

    const int idx = id / n2; // Index of the string considered in str1

    const int idy = id % n2; // Index of the string considered in str2

    if (idx < n1 && idy < n2) {

        // Move the pointer to the first character of the string we are considering

        char *string1 = str1 + offsets1[idx];

        // Computing the length of the string we are considering

        int len1 = offsets1[idx + 1] - offsets1[idx];

        // Move the pointer to the first element of the working memory

        bool *hash_str1 = buffer1 + idy * offsets1[n1] + offsets1[idx];

        char *string2 = str2 + offsets2[idy];

        int len2 = offsets2[idy + 1] - offsets2[idy];

        bool *hash_str2 = buffer2 + idx * offsets2[n2] + offsets2[idy];

        // Compute the Jaro-Winkler Distance between str1[idx] and str2[idy]

        output[id] = jaro_winkler(string1, len1, hash_str1, string2, len2, hash_str2);

    }

}

}
"""

jaro_winkler_kernel = cp.RawKernel(jaro_winkler_code, 'jaro_winkler_kernel')

indices_inverse_code = """
extern "C" {

  __global__ void indices_inverse(long long *input_A,
                                  long long *input_B,
                                  int n_input,
                                  int n_B,
                                  unsigned long long *unique_A_argwhere,
                                  unsigned long long *unique_A_argwhere_offsets,
                                  unsigned int *unique_A_count,
                                  unsigned long long *unique_B_argwhere,
                                  unsigned long long *unique_B_argwhere_offsets,
                                  unsigned int *unique_B_count,
                                  unsigned long long *output,
                                  unsigned long long *output_offsets) {

      const int id = threadIdx.x + blockDim.x * blockIdx.x; // Element of indices being processed

      if (id < n_input) {

        long long id_A = input_A[id];

        long long id_B = input_B[id];

        unsigned int len_A = unique_A_count[id_A]; // Number of observations with id_A in df_A

        unsigned int len_B = unique_B_count[id_B]; // Number of observations with id_B in df_B

        unsigned long long unique_A_off = (id_A == 0 ? 0 : unique_A_argwhere_offsets[id_A - 1]); // Where observations with id_A in df_A start in unique_A_argwhere

        unsigned long long unique_B_off = (id_B == 0 ? 0 : unique_B_argwhere_offsets[id_B - 1]); // Where observations with id_B in df_B start in unique_B_argwhere

        unsigned long long *unique_A_argwhere_off = unique_A_argwhere + unique_A_off; // Offset unique_A_argwhere appropriately

        unsigned long long *unique_B_argwhere_off = unique_B_argwhere + unique_B_off; // Offset unique_B_argwhere appropriately

        int output_off = (id == 0 ? 0 : output_offsets[id - 1]); // Where the output starts in output

        for (int i = 0; i < len_A * len_B; i++) {

          // Transpose indices of pairs in df_A and df_B in output

          output[output_off + i] = unique_A_argwhere_off[i / len_B] * n_B + unique_B_argwhere_off[i % len_B];

        }

      }

    }

}
"""

indices_inverse_kernel = cp.RawKernel(indices_inverse_code, 'indices_inverse')

def jaro_winkler_gpu(str1, str2, offset, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256):
    '''
    This function computes the Jaro-Winkler distance between all pairs of strings in str1 and str2.

    Arguments:
    ----------
    - str1 (NumPy array): First array of strings.
    - str2 (NumPy array): Second array of strings.
    - offset (int): Offset for indices: this value is added to all output indices.

    Optional Arguments:
    -------------------
    - lower_thr (float): Lower threshold for Jaro-Winkler distance.
    - upper_thr (float): Upper threshold for Jaro-Winkler distance.
    - num_threads (int): Number of threads per block. The maximal possible value is 1,024.

    Returns:
    --------
    - output1_gpu (CuPy array): Indices with Jaro-Winkler distance between lower_thr and upper_thr.
    - output2_gpu (CuPy array): Indices with Jaro-Winkler distance above upper_thr.

    The indices represent i * len(str_B) + j, where i is the element's index in str_A and j is the element's index in str_B.
    '''

    mempool = cp.get_default_memory_pool()

    n1 = len(str1) # Number of strings contained in str1

    # Storing strings contained in str1 as an arrow, i.e., characters concatenated next to each other
    str1_arrow = np.frombuffer(''.join(str1).encode(), dtype = np.int8)

    str1_arrow_gpu = cp.array(str1_arrow)

    # Array storing where each string starts and ends: str1[i] begins at offsets[i]
    # and ends at offsets[i + 1] - 1 (inclusively)

    offsets1 = np.append([0], np.cumsum([len(row) for row in str1])).astype(np.int32)

    offsets1_gpu = cp.array(offsets1)

    n2 = len(str2)

    str2_arrow = np.frombuffer(''.join(str2).encode(), dtype = np.int8)

    str2_arrow_gpu = cp.array(str2_arrow)

    offsets2 = np.append([0], np.cumsum([len(row) for row in str2])).astype(np.int32)

    offsets2_gpu = cp.array(offsets2)

    buffer1 = cp.zeros(offsets1[n1] * n2, dtype = bool)

    buffer2 = cp.zeros(offsets2[n2] * n1, dtype = bool)

    output_gpu = cp.zeros(n1 * n2, dtype = cp.float32) # Create output vector

    num_blocks = math.ceil(n1 * n2 / num_threads) # Blocks per Grid

    # Call GPU Kernel
    jaro_winkler_kernel((num_blocks,), (num_threads,), (str1_arrow_gpu, offsets1_gpu, buffer1, n1, str2_arrow_gpu, offsets2_gpu, buffer2, n2, output_gpu))

    # Indices between lower_thr and upper_thr
    indices1_gpu = cp.argwhere(cp.bitwise_and(output_gpu >= lower_thr, output_gpu < upper_thr))

    # Indices above upper_thr
    indices2_gpu = cp.argwhere(output_gpu >= upper_thr)

    # Clean GPU memory
    del str1_arrow, offsets1, buffer1, str2_arrow, offsets2, buffer2, str1_arrow_gpu, offsets1_gpu, str2_arrow_gpu, offsets2_gpu, output_gpu
    mempool.free_all_blocks()

    # Offset indices based on offset parameter
    output1 = cp.ravel(indices1_gpu) + offset

    output2 = cp.ravel(indices2_gpu) + offset

    del indices1_gpu, indices2_gpu
    mempool.free_all_blocks()

    return output1, output2

def jaro_winkler_gpu_unique(str_A, str_B, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256, max_chunk_size = 10000000):
  '''
  This function computes the Jaro-Winkler distance between all pairs of strings in str_A and str_B.

  Arguments:
  ----------
  - str_A (NumPy array): First array of strings.
  - str_B (NumPy array): Second array of strings.

  Optional Arguments:
  -------------------
  - lower_thr (float): Lower threshold for Jaro-Winkler distance.
  - upper_thr (float): Upper threshold for Jaro-Winkler distance.
  - num_threads (int): Number of threads per block. The maximal possible value is 1,024.
  - max_chunk_size (int): Maximal number of pairs per chunk. This value is used to segment the full matrix into chunks.

  Returns:
  --------
  - output1_gpu (CuPy array): Indices with Jaro-Winkler distance between lower_thr and upper_thr.
  - output2_gpu (CuPy array): Indices with Jaro-Winkler distance above upper_thr.

  The indices represent i * len(str_B) + j, where i is the element's index in str_A and j is the element's index in str_B.
  '''

  mempool = cp.get_default_memory_pool()

  # Extracts unique values of str_A (with inverse and counts)
  unique_A, unique_A_inverse, unique_A_counts = np.unique(str_A, return_inverse = True, return_counts = True)

  # This array contains the indices corresponding to each unique value of str_A (as an arrow)
  unique_A_inverse_argsort = np.argsort(unique_A_inverse) # Maybe move to GPU ?

  unique_A_inverse_gpu = cp.array(unique_A_inverse_argsort, dtype = np.uint64)

  # This array contains the number of observations in str_A associated with each unique value
  unique_A_counts_gpu = cp.array(unique_A_counts, dtype = np.uint32)

  # This array contains the offsets necessary to read the indices corresponding to each unique value in str_A
  unique_A_offsets_gpu = cp.cumsum(unique_A_counts_gpu)

  unique_B, unique_B_inverse, unique_B_counts = np.unique(str_B, return_inverse = True, return_counts = True)

  unique_B_inverse_argsort = np.argsort(unique_B_inverse)

  unique_B_inverse_gpu = cp.array(unique_B_inverse_argsort, dtype = np.uint64)

  unique_B_counts_gpu = cp.array(unique_B_counts, dtype = np.uint32)

  unique_B_offsets_gpu = cp.cumsum(unique_B_counts_gpu)

  chunks_A = math.ceil(len(unique_B) / max_chunk_size)

  unique_A_partitions = np.array_split(unique_A, chunks_A)

  unique_A_partitions_len = np.append([0], np.cumsum([len(x) for x in unique_A_partitions]))

  # Compute Jaro-Winkler similarity by chunk

  indices = [jaro_winkler_gpu(x, unique_B, unique_A_partitions_len[i] * len(unique_B), lower_thr, upper_thr, num_threads) for i, x in enumerate(unique_A_partitions)]

  # Concatenate indices of all chunks

  indices1 = cp.concatenate((x[0] for x in indices))

  indices2 = cp.concatenate((x[1] for x in indices))

  del indices
  mempool.free_all_blocks()

  # Inverting indices1, i.e., translate into indices of original dataframes
  indices1_A = indices1 // len(unique_B)

  indices1_B = indices1 % len(unique_B)

  del indices1
  mempool.free_all_blocks()

  output1_count = unique_A_counts_gpu[indices1_A] * unique_B_counts_gpu[indices1_B]

  output1_offsets = cp.cumsum(output1_count)

  output1_gpu = cp.zeros(int(output1_offsets[-1]), dtype = np.uint64)

  num_blocks = math.ceil(len(indices1_A) / num_threads)

  indices_inverse_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, len(indices1_A), len(str_B), unique_A_inverse_gpu, unique_A_offsets_gpu, unique_A_counts_gpu, unique_B_inverse_gpu, unique_B_offsets_gpu, unique_B_counts_gpu, output1_gpu, output1_offsets))

  del indices1_A, indices1_B, output1_count, output1_offsets
  mempool.free_all_blocks()

  # Inverting indices2
  indices2_A = indices2 // len(unique_B)

  indices2_B = indices2 % len(unique_B)

  del indices2
  mempool.free_all_blocks()

  output2_count = unique_A_counts_gpu[indices2_A] * unique_B_counts_gpu[indices2_B]

  output2_offsets = cp.cumsum(output2_count)

  output2_gpu = cp.zeros(int(output2_offsets[-1]), dtype = np.uint64)

  num_blocks = math.ceil(len(indices2_A) / num_threads)

  indices_inverse_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, len(indices2_A), len(str_B), unique_A_inverse_gpu, unique_A_offsets_gpu, unique_A_counts_gpu, unique_B_inverse_gpu, unique_B_offsets_gpu, unique_B_counts_gpu, output2_gpu, output2_offsets))

  del indices2_A, indices2_B, output2_count, output2_offsets, unique_A_inverse_gpu, unique_A_counts_gpu, unique_A_offsets_gpu, unique_B_inverse_gpu, unique_B_counts_gpu, unique_B_offsets_gpu
  mempool.free_all_blocks()

  return output1_gpu, output2_gpu

def merge_indices_pair(indices1, indices2):
  '''
  This function combines two lists of lists of indices. Importantly, it accounts for the fact that one discrete value (or combination thereof) is implicitly ommitted from each list of indices.

  Arguments:
  ----------
  - indices1 (list of CuPy arrays): First list of arrays of indices.
  - indices2 (list of CuPy arrays): First list of arrays of indices.

  Returns:
  --------
  - List of CuPy arrays of indices for all combinations of discrete values of both input lists of arrays of indices. This new list omits the combination formed by the first discrete values of both input lists.
  '''

  mempool = cp.get_default_memory_pool()

  if len(indices1) > 1:
    temp1 = cp.concatenate(indices1) # Indices that do NOT belong to the first discrete value of the first list
  else:
    temp1 = indices1[0]

  if len(indices2) > 1:
    temp2 = cp.concatenate(indices2) # Indices that do NOT belong to the first discrete value of the second list
  else:
    temp2 = indices2[0]

  output = [] # Array used to store the results

  # We iterate over the discrete values of both lists of lists of indices.

  for i in range(len(indices1) + 1):

    # The discrete values of the second list move faster.

    for j in range(len(indices2) + 1):

      if i == 0:

        # We omit the combination formed by the first discrete values of both lists.

        if j != 0:

          output.append(cp.setdiff1d(indices2[j - 1], temp1))

      else:

        if j == 0:

          output.append(cp.setdiff1d(indices1[i - 1], temp2))

        else:

          if len(indices2[j - 1]) > 0:
            output.append(cp.intersect1d(indices1[i - 1], indices2[j - 1]))
          else:
            output.append(cp.empty(0, dtype = np.int64))

  del temp1, temp2
  mempool.free_all_blocks()

  return output

def merge_indices_pair_split(indices1, indices2, max_elements = 2500000):

  mempool = cp.get_default_memory_pool()

  if len(indices1) > 1:
    temp1 = cp.concatenate(indices1) # Indices that do NOT belong to the first discrete value of the first list
  else:
    temp1 = indices1[0]

  if len(indices2) > 1:
    temp2 = cp.concatenate(indices2) # Indices that do NOT belong to the first discrete value of the second list
  else:
    temp2 = indices2[0]

  output = [] # Array used to store the results

  # We iterate over the discrete values of both lists of lists of indices.

  for i in range(len(indices1) + 1):

    # The discrete values of the second list move faster.

    for j in range(len(indices2) + 1):

      if i == 0:

        # We omit the combination formed by the first discrete values of both lists.

        if j != 0:

          chunks_temp1 = math.ceil(len(temp1) / max_elements)
          temp1_split = cp.array_split(temp1, chunks_temp1)

          output_in = functools.reduce(cp.setdiff1d, [indices2[j - 1]] + temp1_split)

          output.append(output_in)

          del temp1_split, output_in
          mempool.free_all_blocks()

      else:

        if j == 0:

          chunks_temp2 = math.ceil(len(temp2) / max_elements)
          temp2_split = cp.array_split(temp2, chunks_temp2)

          output_in = functools.reduce(cp.setdiff1d, [indices1[i - 1]] + temp2_split)

          output.append(output_in)

          del temp2_split, output_in
          mempool.free_all_blocks()

        else:

          if len(indices2[j - 1]) > 0:
            chunks_indices1 = math.ceil(len(indices1[i - 1]) / max_elements)
            indices1_split = cp.array_split(indices1[i - 1], chunks_indices1)

            chunks_indices2 = math.ceil(len(indices2[j - 1]) / max_elements)
            indices2_split = cp.array_split(indices2[j - 1], chunks_indices2)

            output_in = (cp.intersect1d(k, l) for k in indices1_split for l in indices2_split)

            output.append(cp.concatenate(output_in))

            del indices1_split, indices2_split, output_in
            mempool.free_all_blocks()
          else:
            output.append(cp.empty(0, dtype = np.int64))

  del temp1, temp2
  mempool.free_all_blocks()

  return output

def merge_indices(indices):
  '''
  Argument:
  ---------
  - indices (List of lists of CuPy arrays): List of arrays of indices.

  Returns:
  --------
  - List of NumPy arrays of indices.
  '''

  output = functools.reduce(merge_indices_pair_split, indices)

  return output

class Comparison():
  '''
  This class evaluates the similarity between the values in two datasets using the Jaro-Winkler metric.
  '''

  def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, vars_A, vars_B):
    '''
    Arguments:
    ----------
    - df_A (Pandas DataFrame): First dataframe to compare.
    - df_B (Pandas DataFrame): Second dataframe to compare.
    - vars_A (list of str): Names of variables to compare in df_A.
    - vars_B (list of str): Names of variables to compare in df_B. The variables must be listed in the same order as in vars_A.
    '''

    # Check Inputs
    if len(vars_A) != len(vars_B):
      raise Exception('The number of variables in vars_A and vars_B must be the same.')

    if any(var not in df_A.columns for var in vars_A) or any(var not in df_B.columns for var in vars_B):
      raise Exception('The names in vars_A and vars_B must match variables names in df_A and df_B.')

    self.df_A = df_A
    self.df_B = df_B
    self.vars_A = vars_A
    self.vars_B = vars_B
    self._Fit_flag = False

  def fit(self, Lower_Thr = 0.88, Upper_Thr = 0.94, Num_Threads = 256):
    '''
    This method calculates the Jaro-Winkler similarity for every pair of observations across all variables.

    Arguments:
    ----------
    - Lower_Thr (float): Lower threshold for discretizing the Jaro-Winkler similarity.
    - Upper_Thr (float): threshold for discretizing the Jaro-Winkler similarity.
    - Num_Threads (int): Number of threads per block. The maximal possible value is 1,024.

    Sets the Following Attribute:
    -----------------------------
    - Indices (list of CuPy arrays): This list contains the indices of pairs of records in df_A and df_B corresponding to each pattern of discrete levels of similarity across variables. The indices represent i * len(df_B) + j, where i is the element's index in df_A and j is the element's index in df_B.
    '''

    if self._Fit_flag:
      raise Exception('The model has already been fitted.')

    mempool = cp.get_default_memory_pool()
    indices = []

    # Loop over (pairs of) variables and compute the Jaro-Winkler similarity between all pairs of values
    for i in range(len(self.vars_A)):
      indices.append(jaro_winkler_gpu_unique(df_A[self.vars_A[i]].to_numpy(), df_B[self.vars_B[i]].to_numpy(), Lower_Thr, Upper_Thr, Num_Threads))
      mempool.free_all_blocks()

    self.Indices = merge_indices(indices) # Merge discrete levels of similarity over all variables
    self._Fit_flag = True

    del indices
    mempool.free_all_blocks()

  @property
  def Counts(self):
    '''
    Returns:
    --------
    - Counts (NumPy array): This array contains the count of observations for each pattern of discrete levels of similarity across variables.
    '''
    if not self._Fit_flag:
      raise Exception('The model must be fitted first.')

    try:
      return self._Counts
    except:
      counts = [len(x) for x in self.Indices] # Number of pairs for each pattern of discrete levels of similarity
      self._Counts = np.concatenate([[len(self.df_A) * len(self.df_B) - np.sum(counts)], counts]) # Add count of omitted pattern
      return self._Counts
