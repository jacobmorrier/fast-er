import itertools
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorly as tl

class Estimation():
  '''
  This class estimates the parameters of the Fellegi-Sunter model given the observed patterns of discrete levels of similarity across variables.
  '''

  def __init__(self, K, Counts, L = 3):
    '''
    Arguments:
    ----------
    - K (int): Number of variables compared.
    - Counts (NumPy array): This array contains the count of observations for each pattern of discrete levels of similarity across variables.
    - L (int): Number of discrete levels the similarity can take.
    '''

    self.Counts = Counts
    self.K = K # Number of variables
    self.L = L # Number of discrete values (largest across all variables)
    self.Gamma = self._Gamma()
    self._Fit_flag = False

  def _Gamma(self):
    '''
    This internal method generates the representations of all patterns of discrete levels of similarity across variables in the format suitable for Gamma.

    Arguments:
    ----------
    - K (int): Number of variables.
    - L (int): Number of discrete values that can be taken by the variables.

    Sets Attributes:
    ----------------
    - Gamma (Tensor): This three-dimensional tensor encodes all the observed patterns of discrete levels of similarity across variables.
                      The first dimension indexes the patterns.
                      The second dimension represents the variable.
                      The third dimension represents the discrete level of similarity taken by the variable.
    '''

    lb = LabelBinarizer()

    lb.fit(np.eye(self.L))

    Gamma = np.stack([lb.transform(list(l)) for l in list(itertools.product(range(self.L), repeat = self.K))])

    return Gamma

  def _match_probability(self):
    '''
    This internal method computes the conditional match probability for each pattern in Gamma given the current value of the parameters.

    Returns:
    --------
    - Ksi (vector): This vector contains the conditional match probability for each pattern in Gamma.
    '''

    tensor_prob = np.zeros((2, self.Gamma.shape[0]), dtype = np.float32)

    for m in range(2): # Loop over the latent state (i.e., match or no match)

      for k in range(self.K): # Loop over variables

        tensor_prob[m] += np.log(tl.tenalg.mode_dot(self.Gamma[:,k,:], tl.transpose(tl.tensor(self.Pi[k,:,m])), mode = 1))

      tensor_prob[m] = np.exp(tensor_prob[m])

    result = (self.Lambda * tensor_prob[1]) / (self.Lambda * tensor_prob[1] + (1 - self.Lambda) * tensor_prob[0])

    return np.reshape(tl.to_numpy(result), newshape = result.shape[0])

  def fit(self, Tolerance = 1e-4, Max_Iter = 500):
    '''
    This method estimates the parameters of the Fellegi-Sunter model using the Expectation-Maximization (EM) algorithm.

    Arguments:
    ----------
    - Tolerance (float): This parameter governs the convergence of the EM algorithm: convergence is achieved when the largest change in Pi is smaller than the value of this parameter.
    - Max_Iter (int): This parameter determines the maximal number of iterations of the EM algorithm.

    Sets the Following Attributes:
    ------------------------------
    - Lambda (float): Match probability.
    - Pi (Tensor): This three-dimensional vector contains the probability of observing each discrete level of similarity for each variable conditional on the latent state (i.e., match or no match).
                   The first dimension represents the variable.
                   The second dimension represents the discrete level of similarity.
                   The third dimension represents the latent state.
    '''

    if self._Fit_flag:
      raise Exception('The model has already been fitted.')

    # Parameter Initialization
    self.Lambda = np.random.uniform(low = 0, high = 1/2)

    pi_0 = np.random.dirichlet(np.ones(self.L), self.K)
    pi_0 = -np.sort(-pi_0)
    pi_0 = np.transpose(pi_0)

    pi_1 = np.random.dirichlet(np.ones(self.L), self.K)
    pi_1 = np.sort(pi_1)
    pi_1 = np.transpose(pi_1)

    self.Pi = np.stack((pi_0.T, pi_1.T), axis = -1)

    # Proceed with E- and M-steps until convergence
    convergence = False

    iter = 1

    while not convergence and iter <= Max_Iter:

        # E-Step: Compute match probability for possible patterns given current parameters

        ksi = self._match_probability()

        # M-Step: Compute new parameter values consistent with E-step

        self.Lambda = np.dot(ksi, self.Counts) / sum(self.Counts)

        pi_1 = tl.to_numpy(tl.sum(tl.tenalg.batched_outer((self.Gamma, tl.tensor(ksi * self.Counts))), axis = 0)) / np.dot(ksi, self.Counts)
        pi_1 = np.reshape(pi_1, newshape = (self.Gamma.shape[1], self.Gamma.shape[2]))

        pi_0 = tl.to_numpy(tl.sum(tl.tenalg.batched_outer((self.Gamma, tl.tensor((1 - ksi) * self.Counts))), axis = 0)) / np.dot(1 - ksi, self.Counts)
        pi_0 = np.reshape(pi_0, newshape = (self.Gamma.shape[1], self.Gamma.shape[2]))

        new_Pi = np.stack((pi_0, pi_1), axis = -1)

        # Convergence is achieved when the largest change in Pi is smaller than Tolerance
        if np.max(np.absolute(self.Pi - new_Pi)) < Tolerance:
            convergence = True

        self.Pi = new_Pi

        iter += 1

    self._Fit_flag = True

    if convergence:
      print('Convergence successfully achieved.')
    else:
      print('Reached the maximum number of iterations without achieving convergence.')

  @property
  def Ksi(self):
    '''
    This property represents the conditional match probabilities for each pattern of discrete levels of similarity across variables given the estimated parameters of the Fellegi-Sunter model.

    Returns:
    --------
    - Ksi (NumPy array): This array contains the conditional match probabilities for each pattern of discrete levels of similarity across variables.
    '''

    if not self._Fit_flag:
      raise Exception('The model must be fitted first.')

    try:
      return self._Ksi
    except:
      self._Ksi = self._match_probability()
      return self._Ksi
