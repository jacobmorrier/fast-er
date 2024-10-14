import cupy as cp
import numpy as np
import math

_binary_search_code = r"""
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

_binary_search_kernel = cp.RawKernel(_binary_search_code, 'binary_search_kernel')

def intersect(arr1: cp.array, arr2: cp.array, num_threads = 256):
  '''
  This function returns the elements that are common to arr1 and arr2, assuming both arrays are sorted in increasing order.
  
  :param arr1: First array
  :type arr1: cp.array
  :param arr2: Second array
  :type arr2: cp.array
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :return: Array containing elements that are common to both arr1 and arr2
  :rtype: cp.array
  '''

  if arr1.size == 0 or arr2.size == 0:
    return cp.empty(0, dtype = np.uint64)

  mempool = cp.get_default_memory_pool()

  output = cp.zeros(arr1.size, dtype = bool)

  num_blocks = math.ceil(arr1.size / num_threads)

  _binary_search_kernel((num_blocks,), (num_threads,), (arr1, arr2, arr1.size, arr2.size, output, True))

  arr_output = arr1[output]

  del output
  mempool.free_all_blocks()

  return arr_output

def setdiff(arr1: cp.array, arr2: cp.array, num_threads = 256):
  '''
  This function returns the elements that are present in arr1 but not in arr2, assuming both arrays are sorted in increasing order.
  
  :param arr1: First array
  :type arr1: cp.array
  :param arr2: Second array
  :type arr2: cp.array
  :param num_threads: Number of threads per block, defaults to 256
  :type num_threads: int, optional
  :return: Array with elements in arr1 but not in arr2
  :rtype: cp.array
  '''

  if arr1.size == 0 or arr2.size == 0:
    return arr1

  mempool = cp.get_default_memory_pool()

  output = cp.zeros(arr1.size, dtype = bool)

  num_blocks = math.ceil(arr1.size / num_threads)

  _binary_search_kernel((num_blocks,), (num_threads,), (arr1, arr2, arr1.size, arr2.size, output, False))

  arr_output = arr1[output]

  del output
  mempool.free_all_blocks()

  return arr_output

def reduce(function, iterable, initial = None):
  '''
  This function iteratively applies a two-argument function to the elements of an iterable, from left to right, reducing it to a single value. The key difference with functools.reduce is that it releases GPU memory at each step of the process.

  :param function: Two-argument function to be applied to the elements of the iterable
  :param iterable: Iterable
  :type iterable: iterable
  :param initial: Initial value, defaults to None
  :return: Outcome of function applied to the elements of iterable, from left to right
  '''
  
  mempool = cp.get_default_memory_pool()
  
  it = iter(iterable)
  
  if initial is None:
    value = next(it)
  else:
    value = initial
    
  for element in it:
    new_value = function(value, element)
    
    del value
    mempool.free_all_blocks()
    
    value = new_value
    
    del new_value
    mempool.free_all_blocks()
    
  return value
