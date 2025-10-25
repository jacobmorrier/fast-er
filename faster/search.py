import cupy as cp
import numpy as np
import math

_binary_search_code = r"""
extern "C" {

  __device__ bool binary_search(long long x,
                                long long *arr,
                                long long arr_size) {

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

  __global__ void binary_search_kernel(long long *arr1,
                                       long long *arr2,
                                       long long arr1_size,
                                       long long arr2_size,
                                       bool *output,
                                       bool in) {

    const long long id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < arr1_size) {

      output[id] = (in ? binary_search(arr1[id], arr2, arr2_size) : !binary_search(arr1[id], arr2, arr2_size));

    }

  }

}
"""

_binary_search_kernel = cp.RawKernel(_binary_search_code, "binary_search_kernel")

def intersect(arr1: cp.array, arr2: cp.array, num_threads = 256):
  """
  Returns the elements common to ``arr1`` and ``arr2``, assuming both input arrays are sorted in increasing order.
  
  :param arr1: First input array.
  :type arr1: cupy.ndarray
  :param arr2: Second input array.
  :type arr2: cupy.ndarray
  :param num_threads: Number of threads per block. Defaults to 256.
  :type num_threads: int, optional
  :return: Array containing the elements that are common to both ``arr1`` and ``arr2``.
  :rtype: cupy.ndarray
  """

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
  """
  Returns the elements that are present in ``arr1`` but not in ``arr2``, assuming both input arrays are sorted in increasing order.
  
  :param arr1: First input array.
  :type arr1: cupy.ndarray
  :param arr2: Second input array.
  :type arr2: cupy.ndarray
  :param num_threads: Number of threads per block. Defaults to 256.
  :type num_threads: int, optional
  :return: Array containing the elements that are present in ``arr1`` but not in ``arr2``.
  :rtype: cupy.ndarray
  """

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
  """
  Iteratively applies a two-argument function to the elements of an iterable, from left to right, reducing them to a single value. 

  Unlike ``functools.reduce``, this implementation releases GPU memory at each step of the process.

  :param function: A two-argument callable applied cumulatively to the elements of the iterable.
  :type function: callable
  :param iterable: The input iterable whose elements are reduced.
  :type iterable: iterable
  :param initial: Optional initial value to start the reduction. Defaults to None.
  :type initial: any, optional  
  :return: Final outcome obtained after applying ``function`` cumulatively to all elements of the iterable.
  :rtype: any
  """
  
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
