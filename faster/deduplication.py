import cupy as cp
import math
import numpy as np
import pandas as pd
from .search import intersect, setdiff, reduce

jaro_winkler_dedup_code = r"""
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

  __global__ void jaro_winkler_kernel(char *str,
                                      int *length,
                                      long long *offsets,
                                      int n,
                                      bool *buffer1,
                                      long long *offsets1,
                                      bool *buffer2,
                                      long long *offsets2,
                                      float p,
                                      float *output,
                                      int n_output,
                                      int start_row,
                                      int end_row) {

    const long long id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < n_output) {

      const int row = id / n + start_row; // Index of the string processed in str1

      const int col = id % n; // Index of the string processed in str2

      // Only computes Jaro-Winkler similarity if row >= col (preventing redundant comparisons)
      if (row >= col) {

        if (row != col) {

          long long off_row = (row == 0 ? 0 : offsets[row - 1]);

          // Move the pointer to the first character of the string we are processing
          char *string1 = str + off_row;

          // Computing the length of the string we are processing
          int len1 = length[row];

          // Move the pointer to the first element of the working memory
          long long off1 = (row == start_row ? 0 : offsets1[row - start_row - 1]);

          bool *hash_str1 = buffer1 + off1 + len1 * col;

          long long off_col = (col == 0 ? 0 : offsets[col - 1]);

          char *string2 = str + off_col;

          int len2 = length[col];

          long long off2 = (col == 0 ? 0 : offsets2[col - 1]);

          bool *hash_str2 = buffer2 + off2 + len2 * (end_row - 1 - row);

          // Compute the Jaro-Winkler similarity between string1 and string2
          output[id] = jaro_winkler(string1, len1, hash_str1, string2, len2, hash_str2, p);

        } else {

          // A string is identical to itself, so its Jaro-Winkler similarity with itself is 1
          output[id] = 1;

        }

      }

    }

  }

}
"""

jaro_winkler_dedup_kernel = cp.RawKernel(jaro_winkler_dedup_code, 'jaro_winkler_kernel')

output_count_dedup_code = r"""
extern "C" {

  __global__ void output_count(long long *input_A,
                               long long *input_B,
                               int n_input,
                               int *unique_count,
                               int *output) {

    // Element of indices being processed
    const long long id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < n_input) {

      // First input
      long long id_A = input_A[id];

      // Second input
      long long id_B = input_B[id];

      // Number of observations with id_A in df
      int len_A = unique_count[id_A];

      // Number of observations with id_B in df
      int len_B = unique_count[id_B];

      if (id_A != id_B) {

        // Computes the number of pairs of values with id_A and id_B
        output[id] = len_A * len_B;

      } else {

        // If id_A = id_B, we disregard pairs formed by identical elements and those where the row index is less than the column index
        output[id] = len_A * (len_B - 1) / 2;

      }

    }

  }

}
"""

output_count_dedup_kernel = cp.RawKernel(output_count_dedup_code, 'output_count')

indices_inverse_dedup_code = r"""
extern "C" {

  __global__ void indices_inverse(long long *input_A,
                                  long long *input_B,
                                  int n_input,
                                  int n,
                                  long long *unique_argwhere,
                                  int *unique_argwhere_offsets,
                                  int *unique_count,
                                  long long *output,
                                  long long *output_offsets) {

    // Element of indices being processed
    const long long id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < n_input) {

      long long id_A = input_A[id];

      long long id_B = input_B[id];

      int len_A = unique_count[id_A]; // Number of observations with id_A in df_A

      int len_B = unique_count[id_B]; // Number of observations with id_B in df_B

      // Where observations with id_A in df_A start in unique_A_argwhere
      long long unique_A_off = (id_A == 0 ? 0 : unique_argwhere_offsets[id_A - 1]);

      // Where observations with id_B in df_B start in unique_B_argwhere
      long long unique_B_off = (id_B == 0 ? 0 : unique_argwhere_offsets[id_B - 1]);

      // Offset unique_A_argwhere appropriately
      long long *unique_A_argwhere_off = unique_argwhere + unique_A_off;

      // Offset unique_B_argwhere appropriately
      long long *unique_B_argwhere_off = unique_argwhere + unique_B_off;

      // Where the output starts in output
      long long output_off = (id == 0 ? 0 : output_offsets[id - 1]);

      if (id_A != id_B) {

        int k = 0;

        for (int i = 0; i < len_A ; i++) {

          for (int j = 0; j < len_B; j++) {

            // Considers only pairs with the row index greater than the column index

            if (unique_A_argwhere_off[i] > unique_B_argwhere_off[j]) {

              // Transpose indices of pairs in df_A and df_B in output
              output[output_off + k++] = unique_A_argwhere_off[i] * n + unique_B_argwhere_off[j];

            } else {

              // Transpose indices of pairs in df_A and df_B in output
              output[output_off + k++] = unique_B_argwhere_off[j] * n + unique_A_argwhere_off[i];

            }

          }

        }

      } else {

        int k = 0;

        for (int i = 1; i < len_A; i++) {

          // Considers only pairs with the row index greater than the column index

          for (int j = 0; j < i; j++) {

            // Transpose indices of pairs in df_A and df_B in output
            output[output_off + k++] = unique_A_argwhere_off[i] * n + unique_B_argwhere_off[j];

          }

        }

      }

    }

  }

}
"""

indices_inverse_dedup_kernel = cp.RawKernel(indices_inverse_dedup_code, 'indices_inverse')

indices_inverse_exact_dedup_code = r"""
extern "C" {

  __global__ void indices_inverse(long long *input,
                                  int n,
                                  long long *unique_argwhere,
                                  int *unique_argwhere_offsets,
                                  long long *output,
                                  int *output_mask,
                                  int *output_offsets,
                                  int n_output) {

    const long long id = threadIdx.x + blockDim.x * blockIdx.x; // Element of indices being processed

    if (id < n_output) {

      // Input element to which the processed output element refers
      long long mask = output_mask[id];

      // Move pointer to where the output begins in output
      long long output_off = (mask == 0 ? 0 : output_offsets[mask - 1]);

      long long i = id - output_off;

      long long in = input[mask];

      // Row index
      long long row = floorf((sqrtf(8 * i + 1) - 1) / 2);

      // Column index: consider only those lower than row index
      long long col = i - row * (row + 1) / 2;

      long long unique_off = (in == 0 ? 0 : unique_argwhere_offsets[in - 1]);

      long long *unique_argwhere_off = unique_argwhere + unique_off;

      // Transpose indices of pairs in df_A and df_B in output
      output[id] = unique_argwhere_off[row + 1] * n + unique_argwhere_off[col];

    }

  }

}
"""

indices_inverse_exact_dedup_kernel = cp.RawKernel(indices_inverse_exact_dedup_code, 'indices_inverse')

def jaro_winkler_dedup_gpu(string, p = 0.1, lower_thr = 0.88, upper_thr = 0.94, num_threads = 256, max_chunk_size = 2.0):
  """
  This function computes the Jaro-Winkler distance between all unique pairs of values in a string.

  :param string: Array of strings
  :type string: np.array
  :param p: Scaling factor applied to the common prefix in the Jaro-Winkler similarity, defaults to 0.1
  :type p: float, optional
  :param lower_thr: Lower threshold for discretizing Jaro-Winkler distance, defaults to 0.88
  :type lower_thr: float, optional
  :param upper_thr: Upper threshold for discretizing Jaro-Winkler distance, defaults to 0.94
  :type upper_thr: float, optional
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :param max_chunk_size: Maximum memory size per chunk in gigabytes (GB), defaults to 2.0
  :type max_chunk_size: float, optional
  :return: Indices with Jaro-Winkler distance between lower_thr and upper_thr

           Indices with Jaro-Winkler distance above upper_thr

           The indices represent i * len(string) + j, where i is the first element's index and j is the second element's index
  :rtype: [cp.array, cp.array]
  """

  mempool = cp.get_default_memory_pool()

  # Extract unique values of string (with inverse and counts)
  unique, unique_inverse, unique_counts = np.unique(string, return_inverse = True, return_counts = True)

  n_unique = len(unique)

  # Array containing the indices corresponding to each unique value of string (stored as an arrow)
  unique_inverse_gpu = cp.array(unique_inverse, dtype = np.int32)

  unique_inverse_sorted = cp.argsort(unique_inverse_gpu)

  del unique_inverse_gpu
  mempool.free_all_blocks()

  # Array containing the number of observations in string associated with each unique value
  unique_counts_gpu = cp.array(unique_counts, dtype = np.int32)

  # Array containing the offsets necessary to read the indices corresponding to each unique value in string
  unique_offsets_gpu = cp.cumsum(unique_counts_gpu, dtype = np.int32)

  unique_arrow = np.frombuffer(''.join(unique).encode(), dtype = np.int8)

  len_arrow = len(unique_arrow)

  # Array containing the unique values stored as an arrow
  unique_arrow_gpu = cp.array(unique_arrow, dtype = np.int8)

  # Array containing the length of unique values
  unique_len = np.fromiter((len(row) for row in unique), dtype = np.int32, count = len(unique))

  unique_len_gpu = cp.array(unique_len, dtype = np.int32)

  # Array containing the offsets necessary to read the unique values in arrow
  offsets_gpu = cp.cumsum(unique_len_gpu, dtype = np.int64)

  # Approximate the number of chunks required to meet max_chunk_size
  total_comp = len(unique) * (len(unique) + 1) / 2

  chunks = math.ceil((len(unique) * (len(unique) + 1) * 8 + len_arrow * (1 + 2 * len(unique)) + (len(unique) + 1) * 8) / (max_chunk_size * 1024 ** 3 - len_arrow - (len(unique) + 1) * 8))

  # Create partitions accordingly
  chunk_size_row = math.ceil(len(unique) / chunks)

  indices = []

  # Compute the Jaro-Winkler similarity metric by chunk
  for i in range(chunks):

    start_row = i * chunk_size_row

    offset = start_row * len(unique)

    end_row = min((i + 1) * chunk_size_row, len(unique))

    num_comp = end_row * len(unique) - offset

    rows = cp.arange(start_row, end_row, dtype = np.int32)

    # Create working memory for the compute kernel (only for comparisons below the diagonal)
    buffer1_len = unique_len_gpu[rows] * (rows + 1)

    buffer1_offsets = cp.cumsum(buffer1_len, dtype = np.int64)

    del buffer1_len
    mempool.free_all_blocks()

    buffer1 = cp.zeros(int(buffer1_offsets[-1]), dtype = bool)

    if start_row > 0:
      buffer2_len = cp.concatenate((unique_len_gpu[:start_row] * (end_row - start_row), unique_len_gpu[rows] * (end_row - rows)))
    else:
      buffer2_len = unique_len_gpu[rows] * (end_row - rows)

    del rows
    mempool.free_all_blocks()

    buffer2_offsets = cp.cumsum(buffer2_len, dtype = np.int64)

    del buffer2_len
    mempool.free_all_blocks()

    buffer2 = cp.zeros(int(buffer2_offsets[-1]), dtype = bool)

    # Create output vector
    output_gpu = cp.zeros(int(num_comp), dtype = cp.float32)

    # Call the compute kernel on GPU
    num_blocks = math.ceil(num_comp / num_threads)

    jaro_winkler_dedup_kernel((num_blocks,), (num_threads,), (unique_arrow_gpu, unique_len_gpu, offsets_gpu, len(unique), buffer1, buffer1_offsets, buffer2, buffer2_offsets, cp.float32(p), output_gpu, cp.int32(num_comp), cp.int32(start_row), cp.int32(end_row)))

    del buffer1, buffer1_offsets, buffer2, buffer2_offsets
    mempool.free_all_blocks()

    # Extract the indices with Jaro-Winkler similarity between lower_thr and upper_thr
    indices1 = cp.bitwise_and(output_gpu >= lower_thr, output_gpu < upper_thr)

    argwhere1 = cp.argwhere(indices1)

    del indices1
    mempool.free_all_blocks()

    # Extract the indices with Jaro-Winkler similarity above upper_thr
    argwhere2 = cp.argwhere(output_gpu >= upper_thr)

    del output_gpu
    mempool.free_all_blocks()

    # Adjust indices relative to the starting row
    output1 = cp.ravel(argwhere1) + int(offset)

    output2 = cp.ravel(argwhere2) + int(offset)

    del argwhere1, argwhere2
    mempool.free_all_blocks()

    indices.append([output1, output2])

    del output1, output2
    mempool.free_all_blocks()

  del unique_arrow_gpu, unique_len_gpu, offsets_gpu
  mempool.free_all_blocks()

  # Concatenate indices from all chunks
  indices1 = cp.concatenate((x[0] for x in indices), dtype = np.int64)

  indices2 = cp.concatenate((x[1] for x in indices), dtype = np.int64)

  del indices
  mempool.free_all_blocks()

  # Invert indices1, i.e., translate into indices from the original data frame
  indices1_A = indices1 // len(unique)

  indices1_B = indices1 % len(unique)

  del indices1
  mempool.free_all_blocks()

  # Calculate the output count for each input element
  output1_count = cp.zeros(indices1_A.size, dtype = np.int32)

  num_blocks = math.ceil(indices1_A.size / num_threads)

  output_count_dedup_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, indices1_A.size, unique_counts_gpu, output1_count))

  output1_offsets = cp.cumsum(output1_count, dtype = np.int64)

  output1_gpu = cp.zeros(int(output1_offsets[-1]), dtype = np.int64)

  indices_inverse_dedup_kernel((num_blocks,), (num_threads,), (indices1_A, indices1_B, indices1_A.size, len(string), unique_inverse_sorted, unique_offsets_gpu, unique_counts_gpu, output1_gpu, output1_offsets))

  del indices1_A, indices1_B, output1_count, output1_offsets
  mempool.free_all_blocks()

  # Invert indices2
  indices2_A = indices2 // len(unique)

  indices2_B = indices2 % len(unique)

  del indices2
  mempool.free_all_blocks()

  output2_count = cp.zeros(indices2_A.size, dtype = np.int32)

  num_blocks = math.ceil(indices2_A.size / num_threads)

  output_count_dedup_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, indices2_A.size, unique_counts_gpu, output2_count))

  output2_offsets = cp.cumsum(output2_count, dtype = np.int64)

  del output2_count
  mempool.free_all_blocks()

  output2_gpu = cp.zeros(int(output2_offsets[-1]), dtype = np.int64)

  indices_inverse_dedup_kernel((num_blocks,), (num_threads,), (indices2_A, indices2_B, indices2_A.size, len(string), unique_inverse_sorted, unique_offsets_gpu, unique_counts_gpu, output2_gpu, output2_offsets))

  del indices2_A, indices2_B, output2_offsets, unique_inverse_sorted, unique_counts_gpu, unique_offsets_gpu
  mempool.free_all_blocks()

  # Sort output vectors
  output1_sorted = cp.sort(output1_gpu)

  del output1_gpu
  mempool.free_all_blocks()

  output2_sorted = cp.sort(output2_gpu)

  del output2_gpu
  mempool.free_all_blocks()

  return [output1_sorted, output2_sorted]

def exact_dedup_gpu(string, num_threads = 256):
  """
  This function compares all pairs of values in string and returns the indices of pairs that have an exact match

  :param string: Array of strings
  :type string: np.array
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :return: Indices with an exact match

           The indices represent i * len(string) + j, where i is the first element's index and j is the second element's index
  :rtype: [cp.array]
  """

  mempool = cp.get_default_memory_pool()

  # Extract unique values of string (with inverse and counts)
  unique, unique_inverse, unique_counts = np.unique(string, return_inverse = True, return_counts = True)

  # Array containing the indices corresponding to each unique value of string (stored as an arrow)
  unique_inverse_gpu = cp.array(unique_inverse, dtype = np.int64)

  unique_inverse_sorted = cp.argsort(unique_inverse_gpu)

  del unique_inverse_gpu
  mempool.free_all_blocks()

  # Array containing the number of observations in string associated with each unique value
  unique_counts_gpu = cp.array(unique_counts, dtype = np.int32)

  # Array containing the offsets necessary to read the indices corresponding to each unique value in str_A
  unique_offsets_gpu = cp.cumsum(unique_counts_gpu, dtype = np.int32)

  # Extract unique values with at least two frequencies
  indices = cp.argwhere(unique_counts_gpu > 1)

  indices_ravel = cp.ravel(indices)

  del indices
  mempool.free_all_blocks()

  # Invert indices, i.e., translating into indices from original data frame
  output_count = unique_counts_gpu[indices_ravel] * (unique_counts_gpu[indices_ravel] - 1) / 2

  output_offsets = cp.cumsum(output_count, dtype = np.int32)

  # Array indicating for the element of indices to which each element of the output is referring to
  output_mask = cp.repeat(cp.arange(0, indices_ravel.size, dtype = np.int32), repeats = output_count.astype(int).get().tolist())

  output_gpu = cp.zeros(int(output_offsets[-1]), dtype = np.int64)

  num_blocks = math.ceil(output_gpu.size / num_threads)

  indices_inverse_exact_dedup_kernel((num_blocks,), (num_threads,), (indices_ravel, len(string), unique_inverse_sorted, unique_offsets_gpu, output_gpu, output_mask, output_offsets, output_gpu.size))

  del unique_inverse_sorted, unique_counts_gpu, unique_offsets_gpu, indices_ravel, output_count, output_mask, output_offsets
  mempool.free_all_blocks()

  # Sort the output vector
  output_sorted = cp.sort(output_gpu)

  del output_gpu
  mempool.free_all_blocks()

  return [output_sorted]

class Deduplication():
  """
  This class compares the values of selected variables in one dataset.

  :param df: Dataframe to deduplicate
  :type df: pd.DataFrame
  :param Vars_Fuzzy: Names of variables to compare for fuzzy matching in df
  :type Vars_Fuzzy: list of str
  :param Vars_Exact: Names of variables to compare for exact matching in df, defaults to []
  :type Vars_Exact: list of str, optional
  :raises Exception: The variable names in Vars_Fuzzy and Vars_Exact must match variable names in df.
  """

  def __init__(self, df: pd.DataFrame, Vars_Fuzzy, Vars_Exact = []):

    # Check that inputs are valid
    if any(var not in df.columns for var in Vars_Fuzzy) or any(var not in df.columns for var in Vars_Exact):
      raise Exception("The variable names in Vars_Fuzzy and Vars_Exact must match variable names in df.")

    self.df = df
    self.Vars_Fuzzy = Vars_Fuzzy
    self.Vars_Exact = Vars_Exact
    self.Indices = None
    """
    This attribute holds a list of indices corresponding to pairs of records that belong to each pattern of discrete levels of similarity across variables.
    
    :return: Indices for each pattern of discrete levels of similarity across variables

             The indices are calculated as i * len(df) + j, where i is the first element's index and j is the second element's index

             Only pairs for which j is less than i are considered

             Patterns are defined iteratively over variables for fuzzy and exact matching, following the order provided by the user, with the discrete levels of the latter variables moving more quickly

             The pattern with no similiarity is omitted
    :rtype: list of cp.array
    """
    self._Fit_flag = False

  def fit(self, p = 0.1,Lower_Thr = 0.88, Upper_Thr = 0.94, Num_Threads = 256, Max_Chunk_Size = 2.0):
    """
    This method compares all pairs of observations across the selected variables in the dataset. The result is stored in the Indices attribute.

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
    for i in range(len(self.Vars_Fuzzy)):
      indices.append(jaro_winkler_dedup_gpu(self.df[self.Vars_Fuzzy[i]].to_numpy(), p, Lower_Thr, Upper_Thr, Num_Threads, Max_Chunk_Size))
      mempool.free_all_blocks()

    # Loop over variables and compare all pairs of values for exact matching
    for i in range(len(self.Vars_Exact)):
      indices.append(exact_dedup_gpu(self.df[self.Vars_Exact[i]].to_numpy(), Num_Threads))
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
      counts = [x.size for x in self.Indices] # Number of pairs for each pattern of discrete levels of similarity
      self._Counts = np.concatenate([[int(len(self.df) * (len(self.df) + 1) / 2) - np.sum(counts)], counts]) # Add count of omitted pattern
      return self._Counts
