import numpy as np
import matplotlib.pyplot as plt

class Evaluation():
  """
  A class for evaluating the accuracy and uncertainty inherent in the estimates of the Fellegi-Sunter model.

  :param Lambda: Unconditional match probability.
  :type Lambda: float
  :param Ksi: Array containing the conditional match probabilities for each pattern of discrete similarity levels across variables.
  :type Ksi: numpy.ndarray
  :param Counts: Array containing the observed counts for each pattern of discrete similarity levels across the compared variables.
  :type Counts: numpy.ndarray
  """

  def __init__(self, Lambda: float, Ksi: np.array, Counts: np.array):
    self.Lambda = Lambda
    self.Ksi = Ksi
    self.Counts = Counts

  def FDR(self, S: float):
    """
    :param S: Threshold value used to calculate the False Discovery Rate (FDR).
    :type S: float
    :return: The False Discovery Rate (FDR), defined as the proportion of false matches among all pairs with a conditional match probability greater than or equal to the threshold ``S``.
    :rtype: float
    """

    try:
      return np.sum((1 - self.Ksi) * (self.Ksi >= S) * self.Counts) / np.sum((self.Ksi >= S) * self.Counts)
    except:
      return None

  def FNR(self, S: float):
    """
    :param S: Threshold value used to calculate the False Negative Rate (FNR).  
    :type S: float  
    :return: The False Negative Rate (FNR), defined as the proportion of true matches among all pairs with a conditional match probability less than the threshold ``S``.  
    :rtype: float
    """

    try:
      return np.sum(self.Ksi * (self.Ksi < S) * self.Counts) / self.Lambda * np.sum(self.Counts)
    except:
      return None

  def Frontier(self):
    """
    Calculates the False Discovery Rate (FDR) and False Negative Rate (FNR) for all thresholds between 0 and 1 with increments of 1e-3, and displays the resulting frontier curve.
    """

    plt.plot([self.FDR(s / 1000) for s in range(1001)], [self.FNR(s / 1000) for s in range(1001)], ".-")
    plt.xlabel("False Discovery Rate (FDR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.show()

  def Optimal_Threshold(self, Alpha: float):
    """
    Computes the threshold value that minimizes a linear combination of the False Discovery Rate (FDR) and the False Negative Rate (FNR).

    :param Alpha: Weight assigned to the False Negative Rate (FNR) in the linear combination.
    :type Alpha: float
    :return: Threshold value that minimizes the weighted sum of the FDR and FNR.
    :rtype: float
    """

    return np.argmin(np.nan_to_num([self.FDR(s / 1000) + Alpha * self.FNR(s / 1000) for s in range(1001)], nan = 1 + Alpha)) / 1000
