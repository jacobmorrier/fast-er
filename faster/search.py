import cupy as cp
import math

binary_search_code = r"""
extern "C" {

  __device__ bool binary_search(unsigned long long x,
                                unsigned long long *arr,
                                unsigned long long arr_size) {

    long long low = 0;
    long long high = arr_size - 1;

    while (low <= high) {

      long long mid = (low + high) / 2;

      if (arr[mid] == x) {
        return true;
      }

      if (arr[mid] < x) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }

    }

    return false;

  }

  __global__ void binary_search_kernel(unsigned long long *arr1,
                                       unsigned long long *arr2,
                                       unsigned long long arr1_size,
                                       unsigned long long arr2_size,
                                       bool *output,
                                       bool in) {

      const unsigned long long id = threadIdx.x + blockDim.x * blockIdx.x;

      if (id < arr1_size) {

        output[id] = (in ? binary_search(arr1[id], arr2, arr2_size) : !binary_search(arr1[id], arr2, arr2_size));

      }

  }

}
"""

binary_search_kernel = cp.RawKernel(binary_search_code, 'binary_search_kernel')

def intersect(arr1, arr2, num_threads = 256):
  '''
  This function returns the elements that are common between arr1 and arr2, assuming both arrays are sorted in increasing order.
  
  :param arr1: First array.
  :type arr1: cp.array
  :param arr2: Second array.
  :type arr2: cp.array
  :param num_threads: Number of threads per block, defaults to 256.
  :type num_threads: int
  '''

  if arr1.size == 0 or arr2.size == 0:
    return cp.empty(0, dtype = np.uint64)

  mempool = cp.get_default_memory_pool()

  output = cp.zeros(arr1.size, dtype = bool)

  num_blocks = math.ceil(arr1.size / num_threads)

  binary_search_kernel((num_blocks,), (num_threads,), (arr1, arr2, arr1.size, arr2.size, output, True))

  arr_output = arr1[output]

  del output
  mempool.free_all_blocks()

  return arr_output

def setdiff(arr1, arr2, num_threads = 256):
  '''
  This function returns the elements that are present in arr1 but not in arr2, assuming both arrays are sorted in increasing order.
  
  :param arr1: First array.
  :type arr1: cp.array
  :param arr2: Second array.
  :type arr2: cp.array
  :param num_threads: Number of threads per block, defaults to 256.
  :type num_threads: int
  '''

  if arr1.size == 0 or arr2.size == 0:
    return arr1

  mempool = cp.get_default_memory_pool()

  output = cp.zeros(arr1.size, dtype = bool)

  num_blocks = math.ceil(arr1.size / num_threads)

  binary_search_kernel((num_blocks,), (num_threads,), (arr1, arr2, arr1.size, arr2.size, output, False))

  arr_output = arr1[output]

  del output
  mempool.free_all_blocks()

  return arr_output
  
