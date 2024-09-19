import numpy as np
import matplotlib.pyplot as plt

class Evaluation():
  '''
  This class allows us to evaluate the accuracy and uncertainty inherent in the estimates of the Fellegi-Sunter model.
  '''

  def __init__(self, Lambda: float, Ksi: np.array, Counts: np.array):
    '''
    Arguments:
    ----------
    - Lambda (float): Match probability.
    - Ksi (NumPy Array): This array contains the conditional match probabilities for each pattern of discrete levels of similarity across variables.
    - Counts (NumPy Array): This array contains the count of observations for each pattern of discrete levels of similarity across variables.
    '''

    self.Lambda = Lambda
    self.Ksi = Ksi
    self.Counts = Counts

  def FDR(self, S: float):
    '''
    The False Discovery Rate (FDR) represents the proportion of false matches among the pairs for which the conditional match probability is greater than or equal to the threshold S.

    Arguments:
    ----------
    - S (float): Threshold at which the FDR is calculated.

    Returns:
    --------
    - False Discovery Rate (float).
    '''
    try:
      return np.sum((1 - self.Ksi) * (self.Ksi >= S) * self.Counts) / np.sum((self.Ksi >= S) * self.Counts)
    except:
      return None

  def FNR(self, S: float):
    '''
    The false negative rate (FNR) represents the proportion of true matches among the pairs for which the conditional match probability is smaller than the threshold S.

    Arguments:
    ----------
    - S (float): Threshold at which the FNR is calculated.

    Returns:
    --------
    - False Negative Rate (float).
    '''
    try:
      return np.sum(self.Ksi * (self.Ksi < S) * self.Counts) / self.Lambda * np.sum(self.Counts)
    except:
      return None

  def Frontier(self):
    '''
    This method calculates the False Discovery Rate (FDR) and the False Negative Rate (FNR) for all thresholds between 0 and 1 (in steps of 10e-3) and displays the resulting frontier.
    '''

    plt.plot([self.FDR(s / 1000) for s in range(1001)], [self.FNR(s / 1000) for s in range(1001)], '.-')
    plt.xlabel('False Discovery Rate (FDR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.show()

  def Optimal_Threshold(self, Alpha: float):
    '''
    This method computes the value of the threshold that minimizes a linear combination of the False Discovery Rate (FDR) and the False Negative Rate (FNR).

    Arguments:
    ----------
    - Alpha (float): Weight assigned to the False Negative Rate (FNR). The weight assigned to the False Discovery Rate (FDR) is normalized to one.

    Returns:
    --------
    - Threshold that minimizes the linear combination of the FDR and the FNR (float).
    '''

    return np.argmin(np.nan_to_num([self.FDR(s / 1000) + Alpha * self.FNR(s / 1000) for s in range(1001)], nan = 1 + Alpha)) / 1000
