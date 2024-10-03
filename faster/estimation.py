import itertools
import numpy as np

class Estimation():
  '''
  This class estimates the parameters of the Fellegi-Sunter model given the observed patterns of discrete levels of similarity across variables.
  '''

  def __init__(self, K: int, K_Exact: int, Counts: np.array):
    """
    :param K: Number of variables compared for fuzzy matching.
    :type K: int
    :param K_Exact: Number of variables compared for exact matching.
    :type K_Exact: int
    :param Counts: Array containing the count of observations for each pattern of discrete levels of similarity across variables.
    :type Counts: np.array
    """
    
    self.K = K
    self.K_Exact = K_Exact
    self.Counts = Counts
    self.Gamma = self._Gamma()
    self._Fit_flag = False

  def _Gamma(self):
    """
    This internal method generates the representations of all patterns of discrete levels of similarity across variables in the format suitable for Gamma.

    :return: Matrix encoding all the observed patterns of discrete levels of similarity across variables. 
             Each row represents a pattern of discrete levels of similarity. 
             Each column represents a variable. 
             The value of each element represents the discrete level of similarity for a specific variable in a particular pattern.
    :rtype: np.array
    """

    return np.array(list(itertools.product(*(range(i) for i in np.repeat([3,2], [self.K, self.K_Exact])))))

  def _match_probability(self):
    """
    This internal method computes the conditional match probability for each pattern in Gamma given the current value of the parameters.

    :return: Array containing the conditional match probabilities for each pattern of discrete levels of similarity across variables.
    :rtype: np.array
    """

    cond_prob = np.zeros((2, len(self.Gamma)), dtype = np.float32)

    # Loop over latent states
    for m in range(2):
      
      # Loop over variables
      for k in range(self.K):
        
        # Using log-transformation to multiply probabilities of discrete levels of similarity for all variables (conditional on latent variable)
        cond_prob[m,:] += np.log(self.Pi[m][k][self.Gamma[:,k]])

      cond_prob[m,:] = np.exp(cond_prob[m,:])

    # Compute conditional match probability using Bayes' Rule
    result = (self.Lambda * cond_prob[1,:]) / (self.Lambda * cond_prob[1,:] + (1 - self.Lambda) * cond_prob[0,:])

    return result

  def fit(self, Tolerance = 1e-4, Max_Iter = 500):
    """
    This method estimates the parameters of the Fellegi-Sunter model using the Expectation-Maximization (EM) algorithm.
    
    :param Tolerance: Convergence is achieved when the largest change in Pi is smaller than the value of this parameter, defaults to 1e-4.
    :type Tolerance: float, optional
    :param Max_Iter: Maximal number of iterations of the EM algorithm, defaults to 500.
    :type Max_Iter: int, optional
    :raises Exception: If the model has already been fitted, it cannot be fitted again.
    """

    if self._Fit_flag:
      raise Exception('If the model has already been fitted, it cannot be fitted again.')

    # Parameter Initialization
    self.Lambda = np.random.uniform(low = 0, high = 1/2)

    L_by_Variable = np.repeat([3,2], [self.K, self.K_Exact])

    pi_0 = [-np.sort(-np.random.dirichlet(np.ones(i))) for i in L_by_Variable]

    pi_1 = [np.sort(np.random.dirichlet(np.ones(i))) for i in L_by_Variable]

    self.Pi = [pi_0, pi_1]

    # Loop until convergence or the maximum number of iterations is reached
    convergence = False

    iter = 1

    while not convergence and iter <= Max_Iter:

        # E-Step: Compute match probability for possible patterns given current parameters
        ksi = self._match_probability()

        # M-Step: Compute new parameter values consistent with E-step
        self.Lambda = np.dot(ksi, self.Counts) / sum(self.Counts)

        pi_1_denom = np.dot(ksi, self.Counts)
        pi_1 = [np.fromiter((np.dot((self.Gamma[:,k] == l) * self.Counts, ksi) for l in range(L)), dtype = float) / pi_1_denom for k, L in enumerate(L_by_Variable)]

        pi_0_denom = np.dot(1 - ksi, self.Counts)
        pi_0 = [np.fromiter((np.dot((self.Gamma[:,k] == l) * self.Counts, 1 - ksi) for l in range(L)), dtype = float) / pi_0_denom for k, L in enumerate(L_by_Variable)]

        new_Pi = [pi_0, pi_1]

        # Convergence is achieved when the largest change in Pi is smaller than Tolerance
        if np.max(np.absolute(np.concatenate([np.concatenate(x) for x in new_Pi]) - np.concatenate([np.concatenate(x) for x in self.Pi]))) < Tolerance:
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
    """
    This property represents the conditional match probabilities for each pattern of discrete levels of similarity across variables given the estimated parameters of the Fellegi-Sunter model.

    :return: Array containing the conditional match probabilities for each pattern of discrete levels of similarity across variables.
    :rtype: np.array
    :raises Exception: The model must be fitted first.
    """

    if not self._Fit_flag:
      raise Exception('The model must be fitted first.')

    try:
      return self._Ksi
    except:
      self._Ksi = self._match_probability()
      return self._Ksi
