import cupy as cp
import functools
import math
import numpy as np
import pandas as pd
from .comparison import jaro_winkler_gpu

jaro_winkler_dedup_code = r"""
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

        // If there are no matching characters between both strings, the Jaro-Winkler distance between them is 0

        if (match == 0) {

            return 0.0;

        } else {

            // If a positive number of matching characters is found, we need to compute the number of transpositions, that is, the number of matching characters that are not in the right order divided by two

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

            // To go from the Jaro distance to the Jaro-Winkler distance, we need to compute the length of the common prefix between both strings

            int prefix = 0;

            for (int i = 0; i < min(min(len1, len2), 4); i++) {

                if (str1[i] == str2[i]) {

                    prefix++;

                } else {

                    break;

                }

            }

            // To obtain the Jaro-Winkler distance, we adjust the Jaro distance for the length of the common prefix between both strings

            dist += 0.1 * prefix * (1 - dist);

            return dist;

        }

    }

}

__global__ void jaro_winkler_kernel(char *str,
                                    int *offsets,
                                    bool *buffer1,
                                    bool *buffer2,
                                    int n_str,
                                    int offset,
                                    int n,
                                    float *output) {

    // Inputs:
    // - str: First vector of strings stored as an arrow (i.e., concatenated
    //         next to each other)
    // - offsets: Vector storing the index where each string in str starts
    // - buffer1: Working memory to keep track of which characters in the first string are
    //            matching to corresponding characters in the second string
    // - buffer2: Working memory to keep track of which characters in the second string are
    //            matching to corresponding characters in the first string
    // - n_str: Number of strings contained in str
    // - offset: Where to begin in processing strings in str
    // - n: Number of strings to process in str
    // - output: Vector storing the computed Jaro-Winkler distances

    const int id = threadIdx.x + blockDim.x * blockIdx.x;

    const int idx = id / n_str; // Index of the first string considered

    const int idy = id % n_str; // Index of the second string considered

    if (idx < n && offset + idx < n_str && idy < n_str) {

      if (offset + idx != idy) {

        // Move the pointer to the first character of the first string we are considering

        char *string1 = str + offsets[offset + idx];

        // Computing the length of the first string we are considering

        int len1 = offsets[offset + idx + 1] - offsets[offset + idx];

        // Move the pointer to the first element of the working memory

        bool *hash_str1 = buffer1 + idy * (offsets[offset + n] - offsets[offset]) + offsets[offset + idx] - offsets[offset];

        // Move the pointer to the first character of the second string we are considering

        char *string2 = str + offsets[idy];

        // Computing the length of the second string we are considering

        int len2 = offsets[idy + 1] - offsets[idy];

        // Move the pointer to the first element of the working memory

        bool *hash_str2 = buffer2 + idx * offsets[n_str] + offsets[idy];

        // Compute the Jaro-Winkler similarity of str[offset + idx] and str[idy]

        output[id] = jaro_winkler(string1, len1, hash_str1, string2, len2, hash_str2);

      } else {

        // An element is identical to itself (no need to compute JW similarity)

        output[id] = 1;

      }

    }

}

}
"""

jaro_winkler_dedup_kernel = cp.RawKernel(jaro_winkler_dedup_code, 'jaro_winkler_kernel')

output_count_dedup_code = """
extern "C" {

  __global__ void output_count(long long *input_A,
                               long long *input_B,
                               int n_input,
                               unsigned int *unique_count,
                               unsigned int *output) {

      // Element of indices being processed
      const int id = threadIdx.x + blockDim.x * blockIdx.x;

      if (id < n_input) {

        // First input
        long long id_A = input_A[id];

        // Second input
        long long id_B = input_B[id];

        // Number of observations with id_A in df
        unsigned int len_A = unique_count[id_A];

        // Number of observations with id_B in df
        unsigned int len_B = unique_count[id_B];

        if (id_A != id_B) {

          // Computes the number of pairs of values with id_A and id_B
          output[id] = len_A * len_B;

        } else {

          // If id_A = id_B, we must ignore the pairs of values formed by two identical elements
          output[id] = len_A * (len_B - 1);

        }

      }

    }

}
"""

output_count_dedup_kernel = cp.RawKernel(output_count_dedup_code, 'output_count')

indices_inverse_dedup_code = """
extern "C" {

  __global__ void indices_inverse(long long *input_A,
                                  long long *input_B,
                                  int n_input,
                                  int n,
                                  unsigned long long *unique_argwhere,
                                  unsigned long long *unique_argwhere_offsets,
                                  unsigned int *unique_count,
                                  unsigned long long *output,
                                  unsigned long long *output_offsets) {

      const int id = threadIdx.x + blockDim.x * blockIdx.x; // Element of indices being processed

      if (id < n_input) {

        long long id_A = input_A[id];

        long long id_B = input_B[id];

        unsigned int len_A = unique_count[id_A]; // Number of observations with id_A in df_A

        unsigned int len_B = unique_count[id_B]; // Number of observations with id_B in df_B

        unsigned long long unique_A_off = (id_A == 0 ? 0 : unique_argwhere_offsets[id_A - 1]); // Where observations with id_A in df_A start in unique_A_argwhere

        unsigned long long unique_B_off = (id_B == 0 ? 0 : unique_argwhere_offsets[id_B - 1]); // Where observations with id_B in df_B start in unique_B_argwhere

        unsigned long long *unique_A_argwhere_off = unique_argwhere + unique_A_off; // Offset unique_A_argwhere appropriately

        unsigned long long *unique_B_argwhere_off = unique_argwhere + unique_B_off; // Offset unique_B_argwhere appropriately

        unsigned long long output_off = (id == 0 ? 0 : output_offsets[id - 1]); // Where the output starts in output

        if (id_A != id_B) {

          for (int i = 0; i < len_A * len_B; i++) {

            // Transpose indices of pairs in df_A and df_B in output

            output[output_off + i] = unique_A_argwhere_off[i / len_B] * n + unique_B_argwhere_off[i % len_B];

          }

        } else {

          int j = 0;

          for (int i = 0; i < len_A * len_B; i++) {

            // Transpose indices of pairs in df_A and df_B in output

            int row = i / len_B;
            int col = i % len_B;

            if (row != col) {

              output[output_off + j] = unique_A_argwhere_off[row] * n + unique_B_argwhere_off[col];

              j++;

            }

          }

        }

      }

    }

}
"""

indices_inverse_dedup_kernel = cp.RawKernel(indices_inverse_dedup_code, 'indices_inverse')

indices_inverse_exact_dedup_code = """
extern "C" {

  __global__ void indices_inverse(long long *input,
                                  int n,
                                  unsigned long long *unique_argwhere,
                                  unsigned long long *unique_argwhere_offsets,
                                  unsigned int *unique_count,
                                  unsigned long long *output,
                                  long long *output_mask,
                                  unsigned long long *output_offsets,
                                  int n_output) {

      const int id = threadIdx.x + blockDim.x * blockIdx.x; // Element of indices being processed

      if (id < n_output) {

        // The input element to which the processed output element refers
        unsigned long long mask = output_mask[id];

        // Move pointer to where the output begins in output
        unsigned long long output_off = (mask == 0 ? 0 : output_offsets[mask - 1]); 

        unsigned long long i = id - output_off;

        long long in = input[mask];

        unsigned int len = unique_count[in];

        unsigned long long row = i / (len - 1);

        unsigned long long col = i % (len - 1);

        unsigned long long col_adj = (row > col ? col : col + 1);

        unsigned long long unique_off = (in == 0 ? 0 : unique_argwhere_offsets[in - 1]);

        unsigned long long *unique_argwhere_off = unique_argwhere + unique_off;

        output[id] = unique_argwhere_off[row] * n + unique_argwhere_off[col_adj];

      }

  }

}
"""

indices_inverse_exact_dedup_kernel = cp.RawKernel(indices_inverse_exact_dedup_code, 'indices_inverse')

def jaro_winkler_dedup_gpu_unique(string, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256, max_chunk_size = 1):
  """
  This function computes the Jaro-Winkler distance between all pairs of values in string.


  :param string: Array of strings.
  :type string: np.array
  :param lower_thr: Lower threshold for discretizing Jaro-Winkler distance, defaults to 0.88.
  :type lower_thr: float, optional
  :param upper_thr: Upper threshold for discretizing Jaro-Winkler distance, defaults to 0.94.
  :type upper_thr: float, optional
  :param num_threads: Number of threads per block, defaults to 256.
  :type num_threads: int, optional
  :param max_chunk_size: Maximum memory size per chunk in gigabytes (GB), defaults to 1.
  :type max_chunk_size: int, optional
  :return: Indices with Jaro-Winkler distance between lower_thr and upper_thr.
           Indices with Jaro-Winkler distance above upper_thr. 
           The indices represent i * len(string) + j, where i is the first element's index and j is the second element's index.
  :rtype: (cp.array, cp.array)
  """

  mempool = cp.get_default_memory_pool()

  # Extracts unique values of string (with inverse and counts)
  unique, unique_inverse, unique_counts = np.unique(string, return_inverse = True, return_counts = True)

  n_unique = len(unique)

  # This array contains the indices corresponding to each unique value of string (stored as an arrow)
  unique_inverse_argsort = np.argsort(unique_inverse)

  unique_inverse_gpu = cp.array(unique_inverse_argsort, dtype = np.uint64)

  # This array contains the number of observations in string associated with each unique value
  unique_counts_gpu = cp.array(unique_counts, dtype = np.uint32)

  # This array contains the offsets necessary to read the indices corresponding to each unique value in string
  unique_offsets_gpu = cp.cumsum(unique_counts_gpu)

  len_arrow = len(''.join(unique).encode())

  # Determines the number of chunks needed to satisfy max_chunk_size
  chunks = math.ceil((len(unique) ** 2 * 4 + len_arrow * (1 + 2 * len(unique)) + (len(unique) + 1) * 4) / (max_chunk_size * 1024 ** 3 - len_arrow - (len(unique) + 1) * 4))

  # Split array of unique values accordingly
  unique_partitions = np.array_split(unique, chunks)

  unique_partitions_len = np.append([0], np.cumsum([len(x) for x in unique_partitions]))

  # Compute Jaro-Winkler similarity by chunk
  indices = [jaro_winkler_gpu(x, unique, unique_partitions_len[i] * len(unique), lower_thr, upper_thr, num_threads) for i, x in enumerate(unique_partitions)]

  # Concatenate indices of all chunks
  indices1 = cp.concatenate((x[0] for x in indices))

  indices2 = cp.concatenate((x[1] for x in indices))

  del indices
  mempool.free_all_blocks()

  # Inverting indices1, i.e., translate into indices of original data frame
  indices1_A = indices1 // n_unique

  indices1_B = indices1 % n_unique

  del indices1
  mempool.free_all_blocks()

  output1_count = cp.zeros(len(indices1_A), dtype = np.uint32)

  num_blocks = math.ceil(len(indices1_A) / num_threads)

  # Determine the output count for each input element
  output_count_dedup_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, len(indices1_A), unique_counts_gpu, output1_count))

  output1_offsets = cp.cumsum(output1_count)

  output1_gpu = cp.zeros(int(output1_offsets[-1]), dtype = np.uint64)

  indices_inverse_dedup_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, len(indices1_A), len(string), unique_inverse_gpu, unique_offsets_gpu, unique_counts_gpu, output1_gpu, output1_offsets))

  del indices1_A, indices1_B, output1_count, output1_offsets
  mempool.free_all_blocks()

  # Inverting indices2
  indices2_A = indices2 // n_unique

  indices2_B = indices2 % n_unique

  del indices2
  mempool.free_all_blocks()

  output2_count = cp.zeros(len(indices2_A), dtype = np.uint32)

  num_blocks = math.ceil(len(indices2_A) / num_threads)

  output_count_dedup_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, len(indices2_A), unique_counts_gpu, output2_count))

  output2_offsets = cp.cumsum(output2_count)

  output2_gpu = cp.zeros(int(output2_offsets[-1]), dtype = np.uint64)

  indices_inverse_dedup_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, len(indices2_A), len(string), unique_inverse_gpu, unique_offsets_gpu, unique_counts_gpu, output2_gpu, output2_offsets))

  del indices2_A, indices2_B, output2_count, output2_offsets, unique_inverse_gpu, unique_counts_gpu, unique_offsets_gpu
  mempool.free_all_blocks()

  return output1_gpu, output2_gpu

def exact_dedup_gpu(string, num_threads = 256):
  """
  This function compares all pairs of values in string and returns the indices corresponding to the pairs with the same value (i.e., exact match).

  :param string: Array of strings.
  :type string: np.array
  :param num_threads: Number of threads per block, defaults to 256.
  :type num_threads: int, optional
  :return: Indices with an exact match.
           The indices represent i * len(string) + j, where i is the first element's index and j is the second element's index.
  :rtype: (cp.array)
  """

  mempool = cp.get_default_memory_pool()

  # Extracts unique values of string (with inverse and counts)
  unique, unique_inverse, unique_counts = np.unique(string, return_inverse = True, return_counts = True)

  # This array contains the indices corresponding to each unique value of string (stored as an arrow)
  unique_inverse_argsort = np.argsort(unique_inverse)

  unique_inverse_gpu = cp.array(unique_inverse_argsort, dtype = np.uint64)

  # This array contains the number of observations in string associated with each unique value
  unique_counts_gpu = cp.array(unique_counts, dtype = np.uint32)

  # This array contains the offsets necessary to read the indices corresponding to each unique value in str_A
  unique_offsets_gpu = cp.cumsum(unique_counts_gpu)

  # Extract unique values with at least two frequencies
  indices = cp.argwhere(unique_counts_gpu > 1)
  indices_ravel = cp.ravel(indices)

  del indices
  mempool.free_all_blocks()

  # Inverting indices, i.e., translating into indices of original data frame
  output_count = unique_counts_gpu[indices_ravel] * (unique_counts_gpu[indices_ravel] - 1)

  output_offsets = cp.cumsum(output_count)

  # This array indicates for each element of the output, the element of indices it is referring to
  output_mask = cp.repeat(cp.array(range(len(indices_ravel))), repeats = output_count.get().tolist())

  output_gpu = cp.zeros(int(output_offsets[-1]), dtype = np.uint64)

  num_blocks = math.ceil(len(output_gpu) / num_threads)

  indices_inverse_exact_dedup_kernel((num_blocks,), (num_threads,), (indices_ravel, len(string), unique_inverse_gpu, unique_offsets_gpu, unique_counts_gpu, output_gpu, output_mask, output_offsets, len(output_gpu)))

  del unique_inverse_gpu, unique_counts_gpu, unique_offsets_gpu, indices_ravel, output_count, output_mask, output_offsets
  mempool.free_all_blocks()

  return [output_gpu]

def merge_indices(indices, max_elements = 250000):
  """
  This function merges indices across variables to obtain indices corresponding to each pattern of discrete levels of similarity.

  :param indices: List of arrays of indices.
  :type indices: Nested lists of cp.array
  :param max_elements: Maximum number of elements that can be merged or separated simultaneously, defaults to 250000.
  :type max_elements: int, optional
  :return: _description_
  :rtype: list of cp.array
  """

  output = functools.reduce(lambda x, y: merge_indices_pair_split(x, y, max_elements = max_elements), indices)

  return output

class Deduplication():
  """
  This class compares the values of selected variables in one dataset.
  """

  def __init__(self, df: pd.DataFrame, Vars, Vars_Exact = []):
    """

    :param df: Dataframe to deduplicate.
    :type df: pd.DataFrame
    :param Vars: Names of variables to compare for fuzzy matching in df.
    :type Vars: list of strings
    :param Vars_Exact: Names of variables to compare for exact matching in df, defaults to [].
    :type Vars_Exact: list, optional
    :raises Exception: The variable names in Vars and Vars_Exact must match variable names in df.
    """

    # Check that inputs are valid
    if any(var not in df.columns for var in Vars) or any(var not in df.columns for var in Vars_Exact):
      raise Exception('The variable names in Vars and Vars_Exact must match variable names in df.')

    self.df = df
    self.vars = Vars
    self.vars_exact = Vars_Exact
    self._Fit_flag = False

  def fit(self, Lower_Thr = 0.88, Upper_Thr = 0.94, Num_Threads = 256, Max_Chunk_Size = 1, Max_Elements = 250000):
    """
    This method compares all pairs of observations across the selected variables in the dataset.
    It generates a list containing the indices of pairs of records in df_A and df_B that correspond to each pattern of discrete levels of similarity across variables.
    The indices are calculated as i * len(df_B) + j, where i is the first element's index and j is the second element's index.
    
    :param Lower_Thr: Lower threshold for discretizing the Jaro-Winkler similarity, defaults to 0.88.
    :type Lower_Thr: float, optional
    :param Upper_Thr: Upper threshold for discretizing the Jaro-Winkler similarity, defaults to 0.94.
    :type Upper_Thr: float, optional
    :param Num_Threads: Number of threads per block, defaults to 256.
    :type Num_Threads: int, optional
    :param Max_Chunk_Size: Maximum memory size per chunk in gigabytes (GB), defaults to 1.
    :type Max_Chunk_Size: int, optional
    :param Max_Elements: Maximum number of elements that can be merged or separated simultaneously, defaults to 250000.
    :type Max_Elements: int, optional
    :raises Exception: If the model has already been fitted, it cannot be fitted again.
    """

    if self._Fit_flag:
      raise Exception('The model has already been fitted.')

    mempool = cp.get_default_memory_pool()
    indices = []

    # Loop over variables and compute the Jaro-Winkler similarity between all pairs of values
    for i in range(len(self.vars)):
      indices.append(jaro_winkler_dedup_gpu_unique(self.df[self.vars[i]].to_numpy(), Lower_Thr, Upper_Thr, Num_Threads, Max_Chunk_Size))
      mempool.free_all_blocks()

    # Loop over variables and compare all pairs of values for exact matching
    for i in range(len(self.vars_exact)):
      indices.append(exact_dedup_gpu(self.df[self.vars_exact[i]].to_numpy(), Num_Threads))
      mempool.free_all_blocks()

    self.Indices = merge_indices(indices, Max_Elements) # Merge discrete levels of similarity over all variables
    self._Fit_flag = True

    del indices
    mempool.free_all_blocks()

  @property
  def Counts(self):
    """
    _summary_

    :return: An array with the count of observations for each pattern of discrete levels of similarity across variables.
    :rtype: np.array
    :raises Exception: The model must be fitted first.
    """
    if not self._Fit_flag:
      raise Exception('The model must be fitted first.')

    try:
      return self._Counts
    except:
      counts = [len(x) for x in self.Indices] # Number of pairs for each pattern of discrete levels of similarity
      self._Counts = np.concatenate([[len(self.df) * (len(self.df) - 1) - np.sum(counts)], counts]) # Add count of omitted pattern
      return self._Counts
