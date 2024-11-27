import cupy as cp
import math
import numpy as np
import pandas as pd
from .search import intersect, setdiff, reduce
from itertools import accumulate

_jaro_winkler_code = r"""
extern "C"{

  __device__ float jaro_winkler(const char *str1,
                                const int len1,
                                bool *hash_str1,
                                const char *str2,
                                const int len2,
                                bool *hash_str2,
                                float p) {

    // This function computes the Jaro-Winkler similarity between two strings
    // Inputs:
    // - str1: First string
    // - len1: Length of str1
    // - hash_str1: Working memory to keep track of which characters in str1 are
    //              matching to corresponding characters in str2
    // - str2: Second string
    // - len2: Length of str2
    // - hash_str2: Working memory to keep track of which characters in str2 are
    //              matching to corresponding characters in str1
    // - p: Scaling factor applied to the common prefix
    // Output:
    // - dist: Jaro-Winkler similarity between str1 and str2


    if (len1 == 0 || len2 == 0) {

        // If either string is null, the Jaro-Winkler similarity between str1 and str2 is 0
        return 0.0;

    } else {

        // We compute the number of matching characters between str1 and str2

        // We consider the characters max(len1, len2) / 2 - 1 away from each other
        int max_dist = max(len1, len2) / 2 - 1;

        float match = 0;

        for (int i = 0; i < len1; i++) {

            for (int j = max(0, i - max_dist); j < min(len2, i + max_dist + 1); j++) {

                if (str1[i] == str2[j] && hash_str2[j] == false) {

                    // Two characters are matching if they appear in both strings at most max_dist characters away from each other
                    hash_str1[i] = true;
                    hash_str2[j] = true;
                    match++;
                    break;

                }

            }

        }

        if (match == 0) {
        
            // If there is no matching characters between both strings, the Jaro-Winkler similarity between them is 0
            return 0.0;

        } else {

            float t = 0;

            int point = 0;

            // If a positive number of matching characters is found, we need to compute the number of transpositions
            // that is, the number of matching characters that are not in the right order divided by two
            for (int i = 0; i < len1; i++) {

                if (hash_str1[i] == true) {

                    while (hash_str2[point] == false) {

                        point++;

                    }

                    if (str1[i] != str2[point++]) {

                        t++;

                    }

                }

            }

            t /= 2;

            // The Jaro similarity between str1 and str2 is defined as follows:
            float dist = ((match / (float)len1) + (match / (float)len2) + ((match - t) / match)) / 3.0;

            // To go from the Jaro similarity to the Jaro-Winkler similarity, we need
            // to compute the length of the common prefix between both strings
            float prefix = 0;

            for (int i = 0; i < min(min(len1, len2), 4); i++) {

                if (str1[i] == str2[i]) {

                    prefix++;

                } else {

                    break;

                }

            }

            // To obtain the Jaro-Winkler similarity, we adjust the Jaro similarity for the length of the common prefix between both strings
            dist += p * prefix * (1 - dist);

            return dist;

        }

    }

  }

  __global__ void jaro_winkler_kernel(char *str1,
                                      long long *offsets1,
                                      bool *buffer1,
                                      int n1,
                                      char *str2,
                                      long long *offsets2,
                                      bool *buffer2,
                                      int n2,
                                      float p,
                                      float *output) {

    // Inputs:
    // - str1: First array of strings (stored as an arrow)
    // - offsets1: Array storing the index where each string in str1 starts
    // - buffer1: Working memory to keep track of which characters in str1 are
    //            matching to corresponding characters in str2
    // - n1: Number of strings contained in str1
    // - str2: Second array of strings (stored as an arrow)
    // - offsets2: Array storing the index where each string in str2 starts
    // - buffer2: Working memory to keep track of which characters in str2 are
    //            matching to corresponding characters in str1
    // - n2: Number of strings contained in str2
    // - p: Scaling factor applied to the common prefix
    // - output: Array storing the computed Jaro-Winkler similarities

    const long long id = threadIdx.x + blockDim.x * blockIdx.x;

    const long long idx = id / n2; // Index of the string processed in str1

    const long long idy = id % n2; // Index of the string processed in str2

    if (idx < n1 && idy < n2) {

        // Move the pointer to the first character of the string we are processing
        char *string1 = str1 + offsets1[idx];

        // Computing the length of the string we are processing
        int len1 = offsets1[idx + 1] - offsets1[idx];

        // Move the pointer to the first element of the working memory
        bool *hash_str1 = buffer1 + idy * offsets1[n1] + offsets1[idx];

        char *string2 = str2 + offsets2[idy];

        int len2 = offsets2[idy + 1] - offsets2[idy];

        bool *hash_str2 = buffer2 + idx * offsets2[n2] + offsets2[idy];

        // Compute the Jaro-Winkler similarity between str1[idx] and str2[idy]
        output[id] = jaro_winkler(string1, len1, hash_str1, string2, len2, hash_str2, p);

    }

  }

}
"""

_jaro_winkler_kernel = cp.RawKernel(_jaro_winkler_code, "jaro_winkler_kernel")

_indices_inverse_code = r"""
extern "C" {

  __global__ void indices_inverse(long long *input_A,
                                  long long *input_B,
                                  int n_input,
                                  int n_B,
                                  long long *unique_A_argwhere,
                                  int *unique_A_argwhere_offsets,
                                  int *unique_A_count,
                                  long long *unique_B_argwhere,
                                  int *unique_B_argwhere_offsets,
                                  int *unique_B_count,
                                  long long *output,
                                  long long *output_offsets) {

      // Description: This function maps the indices of unique pairs back to their corresponding indices in the original dataframes

      const long long id = threadIdx.x + blockDim.x * blockIdx.x; // Element of indices being processed

      if (id < n_input) {

        long long id_A = input_A[id];

        long long id_B = input_B[id];

        int len_A = unique_A_count[id_A]; // Number of observations with id_A in df_A

        int len_B = unique_B_count[id_B]; // Number of observations with id_B in df_B

        // Where observations with id_A in df_A start in unique_A_argwhere
        int unique_A_off = (id_A == 0 ? 0 : unique_A_argwhere_offsets[id_A - 1]); 

        long long *unique_A_argwhere_off = unique_A_argwhere + unique_A_off; // Offset unique_A_argwhere appropriately

        // Where observations with id_B in df_B start in unique_B_argwhere
        int unique_B_off = (id_B == 0 ? 0 : unique_B_argwhere_offsets[id_B - 1]);

        long long *unique_B_argwhere_off = unique_B_argwhere + unique_B_off; // Offset unique_B_argwhere appropriately

        // Where the output starts in output
        long long output_off = (id == 0 ? 0 : output_offsets[id - 1]); 

        for (int i = 0; i < len_A * len_B; i++) {

          // Transpose indices of pairs in df_A and df_B in output
          output[output_off + i] = unique_A_argwhere_off[i / len_B] * n_B + unique_B_argwhere_off[i % len_B];

      }

    }

  }

}
"""

_indices_inverse_kernel = cp.RawKernel(_indices_inverse_code, "indices_inverse")

def jaro_winkler_gpu(str1, str2, offset = 0, p = 0.1, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256):
  """
  This function computes the Jaro-Winkler similarity between all pairs of strings in str1 and str2.

  :param str1: First array of strings
  :type str1: np.array
  :param str2: Second array of strings
  :type str2: np.array
  :param offset: Value added to all output indices, defaults to 0
  :type offset: int, optional
  :param p: Scaling factor applied to the common prefix in the Jaro-Winkler similarity, defaults to 0.1
  :type p: float, optional
  :param lower_thr: Lower threshold for discretizing the Jaro-Winkler distance, defaults to 0.88
  :type lower_thr: float, optional
  :param upper_thr: Upper threshold for discretizing the Jaro-Winkler distance, defaults to 0.94
  :type upper_thr: float, optional
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :return: Indices with Jaro-Winkler distance between lower_thr and upper_thr
  
           Indices with Jaro-Winkler distance above upper_thr
           
           The indices represent i * len(str_B) + j, where i is the element's index in str_A and j is the element's index in str_B
  :rtype: [cp.array, cp.array]
  """
  
  mempool = cp.get_default_memory_pool()

  n1 = len(str1) # Number of strings contained in str1

  # Storing strings contained in str1 as an arrow, i.e., characters concatenated next to each other
  str1_arrow = np.frombuffer("".join(str1).encode(), dtype = np.int8)

  str1_arrow_gpu = cp.array(str1_arrow, dtype = np.int8)

  # Array storing where each string starts and ends: str1[i] begins at offsets[i] and ends at offsets[i + 1] - 1 (inclusively)
  offsets1 = np.fromiter(accumulate(len(row) for row in str1), dtype = np.int64, count = len(str1))
  offsets1 = np.concatenate(([0], offsets1))

  offsets1_gpu = cp.array(offsets1, dtype = np.int64)

  n2 = len(str2)

  str2_arrow = np.frombuffer("".join(str2).encode(), dtype = np.int8)

  str2_arrow_gpu = cp.array(str2_arrow, dtype = np.int8)

  offsets2 = np.fromiter(accumulate(len(row) for row in str2), dtype = np.int64, count = len(str2))
  offsets2 = np.concatenate(([0], offsets2))
  
  offsets2_gpu = cp.array(offsets2, dtype = np.int64)

  # Create working memory
  buffer1 = cp.zeros(offsets1[n1] * n2, dtype = bool)

  buffer2 = cp.zeros(offsets2[n2] * n1, dtype = bool)

  output_gpu = cp.zeros(n1 * n2, dtype = cp.float32) # Create output vector

  num_blocks = math.ceil(n1 * n2 / num_threads) # Blocks per grid

  # Call GPU Kernel
  _jaro_winkler_kernel((num_blocks,), (num_threads,), (str1_arrow_gpu, offsets1_gpu, buffer1, n1, str2_arrow_gpu, offsets2_gpu, buffer2, n2, cp.float32(p), output_gpu))

  # Clean GPU memory
  del str1_arrow, offsets1, buffer1, str2_arrow, offsets2, buffer2, str1_arrow_gpu, offsets1_gpu, str2_arrow_gpu, offsets2_gpu
  mempool.free_all_blocks()
  
  # Indices between lower_thr and upper_thr
  indices1 = cp.bitwise_and(output_gpu >= lower_thr, output_gpu < upper_thr)
  
  argwhere1 = cp.argwhere(indices1)
  
  del indices1
  mempool.free_all_blocks()

  # Indices above upper_thr
  argwhere2 = cp.argwhere(output_gpu >= upper_thr)

  # Clean GPU memory
  del output_gpu
  mempool.free_all_blocks()

  # Offset indices based on parameter
  output1 = cp.ravel(argwhere1) + offset

  output2 = cp.ravel(argwhere2) + offset

  # Clean GPU memory
  del argwhere1, argwhere2
  mempool.free_all_blocks()

  return [output1, output2]

def jaro_winkler_unique_gpu(str_A, str_B, p = 0.1, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256, max_chunk_size = 2.0):
  """
  This function computes in chunks the Jaro-Winkler similarity between all pairs of strings in str_A and str_B. To speed up processing, this function restricts comparisons to unique values in both input strings.

  :param str_A: First array of strings
  :type str_A: np.array
  :param str_B: Second array of strings
  :type str_B: np.array
  :param p: Scaling factor applied to the common prefix in the Jaro-Winkler similarity, defaults to 0.1
  :type p: float, optional
  :param lower_thr: Lower threshold for discretizing the Jaro-Winkler distance, defaults to 0.88
  :type lower_thr: float, optional
  :param upper_thr: Upper threshold for discretizing the Jaro-Winkler distance, defaults to 0.94
  :type upper_thr: float, optional
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :param max_chunk_size: Maximum memory size per chunk in gigabytes (GB), defaults to 2.0
  :type max_chunk_size: float, optional
  :return: Indices with Jaro-Winkler distance between lower_thr and upper_thr
  
           Indices with Jaro-Winkler distance above upper_thr
           
           The indices represent i * len(str_B) + j, where i is the element's index in str_A and j is the element's index in str_B
  :rtype: [cp.array, cp.array]
  """

  mempool = cp.get_default_memory_pool()

  # Extracts unique values of str_A (with inverse and counts)
  unique_A, unique_A_inverse, unique_A_counts = np.unique(str_A, return_inverse = True, return_counts = True)

  # Array containing the indices corresponding to each unique value of str_A (as an arrow)
  unique_A_inverse_gpu = cp.array(unique_A_inverse, dtype = np.int32)
  
  unique_A_inverse_sorted = cp.argsort(unique_A_inverse_gpu)

  del unique_A_inverse_gpu
  mempool.free_all_blocks()
  
  # Array containing the number of observations in str_A associated with each unique value
  unique_A_counts_gpu = cp.array(unique_A_counts, dtype = np.int32)

  # Array containing the offsets necessary to read the indices corresponding to each unique value in str_A
  unique_A_offsets_gpu = cp.cumsum(unique_A_counts_gpu, dtype = np.int32)

  len_A_arrow = len("".join(unique_A).encode()) # Length of arrow (for approximation of the number of chunks)

  unique_B, unique_B_inverse, unique_B_counts = np.unique(str_B, return_inverse = True, return_counts = True)

  unique_B_inverse_gpu = cp.array(unique_B_inverse, dtype = np.int32)
  
  unique_B_inverse_sorted = cp.argsort(unique_B_inverse_gpu)

  del unique_B_inverse_gpu
  mempool.free_all_blocks()

  unique_B_counts_gpu = cp.array(unique_B_counts, dtype = np.int32)

  unique_B_offsets_gpu = cp.cumsum(unique_B_counts_gpu, dtype = np.int32)

  len_B_arrow = len("".join(unique_B).encode())

  # Approximate the number of chunks needed to satisfy max_chunk_size
  chunks = math.ceil((len(unique_A) * len(unique_B) * 4 + len_A_arrow * (1 + len(unique_B)) + len_B_arrow * (1 + len(unique_A)) + (len(unique_A) + 1) * 8 + (len(unique_B) + 1) * 8) / (max_chunk_size * 1024 ** 3 - len_B_arrow - (len(unique_B) + 1) * 8))

  # Split array of unique values accordingly
  unique_A_partitions = np.array_split(unique_A, chunks)

  unique_A_partitions_len = np.append([0], np.cumsum([len(x) for x in unique_A_partitions]))

  # Compute Jaro-Winkler similarity in chunks
  indices = [jaro_winkler_gpu(x, unique_B, unique_A_partitions_len[i] * len(unique_B), p, lower_thr, upper_thr, num_threads) for i, x in enumerate(unique_A_partitions)]

  # Concatenate indices of all chunks
  indices1 = cp.concatenate((x[0] for x in indices), dtype = np.int64)

  indices2 = cp.concatenate((x[1] for x in indices), dtype = np.int64)

  del indices
  mempool.free_all_blocks()

  if indices1.size > 0:
  
    # Inverting indices1, i.e., translate into indices of original dataframes
    indices1_A = indices1 // len(unique_B) # Unique values in df_A
  
    indices1_B = indices1 % len(unique_B) # Unique values in df_B
  
    del indices1
    mempool.free_all_blocks()
  
    # Counts of indices from original dataframes corresponding to each index from unique values
    output1_count = unique_A_counts_gpu[indices1_A] * unique_B_counts_gpu[indices1_B] 
  
    # Array containing where indices from original dataframes start in output for each index from unique values
    output1_offsets = cp.cumsum(output1_count, dtype = np.int64) 
  
    # Create output vector
    output1_gpu = cp.zeros(int(output1_offsets[-1]), dtype = np.int64) 
  
    num_blocks = math.ceil(indices1_A.size / num_threads)
  
    _indices_inverse_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, indices1_A.size, len(str_B), unique_A_inverse_sorted, unique_A_offsets_gpu, unique_A_counts_gpu, unique_B_inverse_sorted, unique_B_offsets_gpu, unique_B_counts_gpu, output1_gpu, output1_offsets))
  
    del indices1_A, indices1_B, output1_count, output1_offsets
    mempool.free_all_blocks()

    # Sort output vectors
    output1_sorted = cp.sort(output1_gpu)
    del output1_gpu
    mempool.free_all_blocks()

  else:

    output1_sorted = cp.zeros(0, dtype = np.int64)

  if indices2.size > 0:
    
    # Inverting indices2
    indices2_A = indices2 // len(unique_B)
  
    indices2_B = indices2 % len(unique_B)
  
    del indices2
    mempool.free_all_blocks()
  
    output2_count = unique_A_counts_gpu[indices2_A] * unique_B_counts_gpu[indices2_B]
  
    output2_offsets = cp.cumsum(output2_count, dtype = np.int64)
  
    output2_gpu = cp.zeros(int(output2_offsets[-1]), dtype = np.int64)
  
    num_blocks = math.ceil(indices2_A.size / num_threads)
  
    _indices_inverse_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, indices2_A.size, len(str_B), unique_A_inverse_sorted, unique_A_offsets_gpu, unique_A_counts_gpu, unique_B_inverse_sorted, unique_B_offsets_gpu, unique_B_counts_gpu, output2_gpu, output2_offsets))
  
    del indices2_A, indices2_B, output2_count, output2_offsets, unique_A_inverse_sorted, unique_A_counts_gpu, unique_A_offsets_gpu, unique_B_inverse_sorted, unique_B_counts_gpu, unique_B_offsets_gpu
    mempool.free_all_blocks()
  
    output2_sorted = cp.sort(output2_gpu)
    del output2_gpu
    mempool.free_all_blocks()

  else:

    output2_sorted = cp.zeros(0, dtype = np.int64)

    del unique_A_inverse_sorted, unique_A_counts_gpu, unique_A_offsets_gpu, unique_B_inverse_sorted, unique_B_counts_gpu, unique_B_offsets_gpu
    mempool.free_all_blocks()

  return [output1_sorted, output2_sorted]

def exact_gpu(str_A, str_B, num_threads = 256):
  """
  This function compares all pairs of values in str_A and str_B and returns the indices of pairs with the same value (i.e., exact match).
  
  :param str_A: First array of strings
  :type str_A: np.array
  :param str_B: Second array of strings
  :type str_B: np.array
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :return: Indices with an exact match
  
           The indices represent i * len(str_B) + j, where i is the element's index in str_A and j is the element's index in str_B
  :rtype: [cp.array]
  """

  mempool = cp.get_default_memory_pool()

  # Extracts unique values of str_A (with inverse and counts)
  unique_A, unique_A_inverse, unique_A_counts = np.unique(str_A, return_inverse = True, return_counts = True)

  # This array contains the indices corresponding to each unique value of str_A (as an arrow)
  unique_A_inverse_gpu = cp.array(unique_A_inverse, dtype = np.int32)

  unique_A_inverse_sorted = cp.argsort(unique_A_inverse_gpu)

  del unique_A_inverse_gpu
  mempool.free_all_blocks()

  # This array contains the number of observations in str_A associated with each unique value
  unique_A_counts_gpu = cp.array(unique_A_counts, dtype = np.int32)

  # This array contains the offsets necessary to read the indices corresponding to each unique value in str_A
  unique_A_offsets_gpu = cp.cumsum(unique_A_counts_gpu, dtype = np.int32)

  unique_B, unique_B_inverse, unique_B_counts = np.unique(str_B, return_inverse = True, return_counts = True)

  unique_B_inverse_gpu = cp.array(unique_B_inverse, dtype = np.int32)

  unique_B_inverse_sorted = cp.argsort(unique_B_inverse_gpu)

  del unique_B_inverse_gpu
  mempool.free_all_blocks()

  unique_B_counts_gpu = cp.array(unique_B_counts, dtype = np.int32)

  unique_B_offsets_gpu = cp.cumsum(unique_B_counts_gpu, dtype = np.int32)

  unique_all, unique_all_inverse, unique_all_counts = np.unique(np.concatenate((unique_A, unique_B)), return_inverse = True, return_counts = True)

  unique_all_inverse_gpu = cp.array(unique_all_inverse, dtype = np.int32)

  unique_all_inverse_argsort = cp.argsort(unique_all_inverse_gpu)

  unique_all_counts_gpu = cp.array(unique_all_counts, dtype = np.int32)

  unique_all_offsets_gpu = cp.cumsum(unique_all_counts_gpu, dtype = np.int32)

  # The values in both unique_A and unique_B have a count of 2
  equal_indices = cp.argwhere(unique_all_counts_gpu == 2)
  
  equal_indices_raveled = cp.ravel(equal_indices)
  
  del equal_indices
  mempool.free_all_blocks()

  if equal_indices_raveled.size > 0:

    indices_A = unique_all_inverse_argsort[unique_all_offsets_gpu[equal_indices_raveled] - 2]
  
    indices_B = unique_all_inverse_argsort[unique_all_offsets_gpu[equal_indices_raveled] - 1] - len(unique_A)
  
    del unique_all_inverse_gpu, unique_all_inverse_argsort, unique_all_counts_gpu, unique_all_offsets_gpu, equal_indices_raveled
    mempool.free_all_blocks()
  
    output_count = unique_A_counts_gpu[indices_A] * unique_B_counts_gpu[indices_B]
  
    output_offsets = cp.cumsum(output_count, dtype = np.int64)
  
    output_gpu = cp.zeros(int(output_offsets[-1]), dtype = np.int64)
  
    num_blocks = math.ceil(indices_A.size / num_threads)
  
    _indices_inverse_kernel((num_blocks,), (num_threads,), (indices_A, indices_B, indices_A.size, len(str_B), unique_A_inverse_sorted, unique_A_offsets_gpu, unique_A_counts_gpu, unique_B_inverse_sorted, unique_B_offsets_gpu, unique_B_counts_gpu, output_gpu, output_offsets))
  
    del indices_A, indices_B, output_count, output_offsets, unique_A_inverse_sorted, unique_A_counts_gpu, unique_A_offsets_gpu, unique_B_inverse_sorted, unique_B_counts_gpu, unique_B_offsets_gpu
    mempool.free_all_blocks()
  
    output_sorted = cp.sort(output_gpu)
    
    del output_gpu
    mempool.free_all_blocks()

  else:

    output_sorted = cp.zeros(0, dtype = np.int64)

    del unique_A_inverse_sorted, unique_A_counts_gpu, unique_A_offsets_gpu, unique_B_inverse_sorted, unique_B_counts_gpu, unique_B_offsets_gpu
    mempool.free_all_blocks()

  return [output_sorted]

class Comparison():
  """
  This class compares the values of selected variables in two data frames.

  :param df_A: First data frame to compare
  :type df_A: pd.DataFrame
  :param df_B: Second data frame to compare
  :type df_B: pd.DataFrame
  :param Vars_Fuzzy_A: Names of variables to compare for fuzzy matching in df_A
  :type Vars_Fuzzy_A: list of str
  :param Vars_Fuzzy_B: Names of variables to compare for fuzzy matching in df_B listed in the same order as in Vars_Fuzzy_A
  :type Vars_Fuzzy_B: list of str
  :param Vars_Exact_A: Names of variables to compare for exact matching in df_A, defaults to []
  :type Vars_Exact_A: list of str, optional
  :param Vars_Exact_B: Names of variables to compare for exact matching in df_B listed in the same order as in Vars_Exact_A, defaults to []
  :type Vars_Exact_B: list of str, optional
  :raises Exception: The lengths of Vars_Fuzzy_A and Vars_Fuzzy_B must be the same.
  :raises Exception: The lengths of Vars_Exact_A and Vars_Exact_B must be the same.
  :raises Exception: The names in Vars_Fuzzy_A and Vars_Fuzzy_B must match variables names in df_A and df_B.
  :raises Exception: The names in Vars_Exact_A and Vars_Exact_B must match variables names in df_A and df_B.
  """

  def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, Vars_Fuzzy_A, Vars_Fuzzy_B, Vars_Exact_A = [], Vars_Exact_B = []):

    # Check Inputs
    if len(Vars_Fuzzy_A) != len(Vars_Fuzzy_B):
      raise Exception("The lengths of Vars_Fuzzy_A and Vars_Fuzzy_B must be the same.")

    if len(Vars_Exact_A) != len(Vars_Exact_B):
      raise Exception("The lengths of Vars_Exact_A and Vars_Exact_B must be the same.")

    if any(var not in df_A.columns for var in Vars_Fuzzy_A) or any(var not in df_B.columns for var in Vars_Fuzzy_B):
      raise Exception("The names in Vars_Fuzzy_A and Vars_Fuzzy_B must match variables names in df_A and df_B.")

    if any(var not in df_A.columns for var in Vars_Exact_A) or any(var not in df_B.columns for var in Vars_Exact_B):
      raise Exception("The names in Vars_Exact_A and Vars_Exact_B must match variables names in df_A and df_B.")

    self.df_A = df_A
    self.df_B = df_B
    self.Vars_Fuzzy_A = Vars_Fuzzy_A
    self.Vars_Fuzzy_B = Vars_Fuzzy_B
    self.Vars_Exact_A = Vars_Exact_A
    self.Vars_Exact_B = Vars_Exact_B
    self.Indices = None
    """
    This attribute holds a list of indices corresponding to pairs of records in df_A and df_B that belong to each pattern of discrete levels of similarity across variables.
    
    :return: Indices for each pattern of discrete levels of similarity across variables

             The indices are calculated as i * len(df_B) + j, where i is the element's index in df_A and j is the element's index in df_B

             Patterns are defined iteratively over variables for fuzzy and exact matching, following the order provided by the user, with the discrete levels of the latter variables moving more quickly

             The pattern with no similiarity is omitted
    :rtype: list of cp.array
    """
    self._Fit_flag = False

  def fit(self, p = 0.1, Lower_Thr = 0.88, Upper_Thr = 0.94, Num_Threads = 256, Max_Chunk_Size = 2.0):
    """
    This method compares all pairs of observations across the selected variables in both data frames. The result is stored in the Indices attribute.

    :param p: Scaling factor applied to the common prefix in the Jaro-Winkler similarity, defaults to 0.1
    :type p: float, optional
    :param Lower_Thr: Lower threshold for discretizing the Jaro-Winkler similarity, defaults to 0.88
    :type Lower_Thr: float, optional
    :param Upper_Thr: Upper threshold for discretizing the Jaro-Winkler similarity, defaults to 0.94
    :type Upper_Thr: float, optional
    :param Num_Threads: Number of threads per block, defaults to 256
    :type Num_Threads: int, optional
    :param Max_Chunk_Size: Maximum memory size per chunk in gigabytes (GB), defaults to 2.0
    :type Max_Chunk_Size: float, optional
    :raises Exception: If the model has already been fitted, it cannot be fitted again.
    """

    if self._Fit_flag:
      raise Exception("If the model has already been fitted, it cannot be fitted again.")

    mempool = cp.get_default_memory_pool()
    indices = []

    # Loop over variables and compute the Jaro-Winkler similarity between all pairs of values
    for i in range(len(self.Vars_Fuzzy_A)):
      indices.append(jaro_winkler_unique_gpu(self.df_A[self.Vars_Fuzzy_A[i]].to_numpy(), self.df_B[self.Vars_Fuzzy_B[i]].to_numpy(), p, Lower_Thr, Upper_Thr, Num_Threads, Max_Chunk_Size))
      mempool.free_all_blocks()

    # Loop over variables and compare all pairs of values for exact matching
    for i in range(len(self.Vars_Exact_A)):
      indices.append(exact_gpu(self.df_A[self.Vars_Exact_A[i]].to_numpy(), self.df_B[self.Vars_Exact_B[i]].to_numpy(), Num_Threads))
      mempool.free_all_blocks()

    # Merge discrete levels of similarity over all variables
    self.Indices = indices[0]
    del indices[0]
    mempool.free_all_blocks()

    while len(indices) > 0:

      output = []

      for j in range(len(indices[0])):

        output.append(reduce(setdiff, self.Indices, indices[0][j]))
        mempool.free_all_blocks()

      while len(self.Indices) > 0:

        output.append(reduce(setdiff, indices[0], self.Indices[0]))
        mempool.free_all_blocks()

        for j in range(len(indices[0])):

          output.append(intersect(self.Indices[0], indices[0][j]))
          mempool.free_all_blocks()

        del self.Indices[0]
        mempool.free_all_blocks()

      self.Indices = output

      del indices[0], output
      mempool.free_all_blocks()
      
    self._Fit_flag = True

    del indices
    mempool.free_all_blocks()

  @property
  def Counts(self):
    """
    This property holds the count of observations for each pattern of discrete levels of similarity across variables.
    
    :return: Array with the count of observations for each pattern of discrete levels of similarity across variables
    :rtype: np.array
    """
    if not self._Fit_flag:
      raise Exception("The model must be fitted first.")

    try:
      return self._Counts
    except:
      counts = [x.size for x in self.Indices] # Count of pairs for each pattern of discrete levels of similarity
      self._Counts = np.concatenate([[len(self.df_A) * len(self.df_B) - np.sum(counts)], counts]) # Add count of omitted pattern
      return self._Counts
