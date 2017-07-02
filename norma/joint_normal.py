import copy
import itertools

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy import stats


class JointNormal(object):
    """
    :type labels: list
    :param labels: A list of string labels for the variables in this distribution

    :type mu: iterable of numbers or N x 1 numpy.matrix
    :param mu: The mean vector

    :type cov: list of lists or numpy.matrix
    :param cov: a list containing the rows of the covariance matrix

    :type N:  int or float
    :param N: The number of observations that went into this distribution.  This is used
              for weighting the relative importance of new observations added through the
              :code:`.ingest()` method.

    :type n_max:  int
    :param n_max: The maximum value N can have.  Limiting N has the effect of allowing
                  the distrubtion to "forget" old measurements.

    This class provides an abstraction for creating, manipulating and querying multi-variate
    normal distributions.

    """
    def __init__(self, labels, mu=None, cov=None, N=0., n_max=None):
        dim = len(labels)

        # private attributes holding
        # _mu: mean vector
        # _cov: covariance matrix
        self._mu = None
        self._cov = None

        # N: the number of observations used to generate the current distribution
        self.N = N

        # n_max: cap the number of observations at this value to enable forgetting
        self.n_max = n_max

        self.labels = [ll for ll in labels]
        self._index_for_label = {label: ind for (ind, label) in enumerate(labels)}

        # use setters after labels have been defined to catch misshaped inputs
        self.mu = self._vectorize(np.zeros(dim)) if mu is None else mu
        self.cov = np.matrix(np.zeros((dim, dim))) if cov is None else cov

    @property
    def mu(self):
        """
        :rtype: numpy.matrix
        :return: N x 1 mean vector
        """
        return copy.deepcopy(self._mu)

    @mu.setter
    def mu(self, mu):
        """
        :type mu: iterable
        :param mu: assign mu to this list of numbers
        """
        if len(mu) != len(self.labels):
            raise ValueError('Setting mu with wrong dimensions')
        self._mu = self._vectorize(mu)

    @property
    def cov(self):
        """
        :rtype: numpy.matrix
        :return: N x N covariance matrix
        """
        return copy.deepcopy(self._cov)

    @cov.setter
    def cov(self, cov):
        """
        :type cov: list of lists or numpy matrix
        :param cov: list of covariance matrix rows or an N x N numpy matrix
        """
        dim = len(self.labels)
        new_cov = np.matrix(cov)
        if new_cov.shape != (dim, dim):
            raise ValueError('Setting covariance with wrong dimensions')
        self._cov = new_cov

    @property
    def mu_cov_n(self):
        """
        :rtype: `tuple (numpy.matrix, numpy.matrix, int)`
        :return: `(mu, cov, N)`
        """
        return (self._mu, self._cov, self.N)

    def _index_for(self, labels):
        bad_labels = set(labels) - set(self.labels)
        if len(bad_labels) > 0:
            raise ValueError('Unrecognized labels: {}'.format(bad_labels))
        return [self._index_for_label[label] for label in labels]

    def _labels_for(self, index):
        return [self.labels[ind] for ind in index]

    @property
    def mu_frame(self):
        """
        :rtype: pandas.Dataframe
        :return: Dataframe holding mean vector in column labeled 'mu'
        """
        return pd.DataFrame(self._mu[:, 0], index=self.labels, columns=['mu'])

    @property
    def cov_frame(self):
        """
        :rtype: pandas.Dataframe
        :return: Dataframe holding covariance matrix with columns and index holding variable names
        """
        return pd.DataFrame(self.cov, index=self.labels, columns=self.labels)

    def _validate_frame(self, df):
        missing_vars = set(self.labels) - set(df.columns)
        if missing_vars:
            raise ValueError('Input dataframe missing columns {}'.format(missing_vars))

    def _vectorize(self, x):
        return np.matrix(np.reshape(x, (len(x), 1)))

    def _update_cov(self, x, mu, cov, n):
        # x: a vector of observations
        # mu: the mean vector
        # cov: the covariance matrix
        # k: the "kalman gain" constant
        # n: the number of observations observed before the update
        k = 1. / (n + 1.)

        cov_prime = cov + k * (x - mu) * (x - mu).T
        return (1. - k) * cov_prime

    def _update_mu(self, x, mu, n):
        # x: a vector of observations
        # mu: the mean vector
        # n: the number of observations observed before the update
        # k: the "kalman gain" constant
        k = 1. / (n + 1.)
        return mu + k * (x - mu)

    def _update_n(self, n, n_max=None):
        # n: the number of observations observed before the update
        if self.n_max and n >= self.n_max:
            return self.n_max
        else:
            return n + 1

    def _ingest_point(self, x):
        # x: a vector of observations
        # mu: the mean vector
        # cov: the covariance matrix
        # N: the number of observations
        mu = self._update_mu(x, self._mu, self.N)
        cov = self._update_cov(x, self._mu, self._cov, self.N)
        N = self._update_n(self.N)

        # save new state  (must compute this in two steps to avoid
        #                  ensure that all updated values are computed
        #                  using previous values)
        self._mu, self._cov, self.N = mu, cov, N

    def ingest(self, data):
        """
        :type data: valid argument to pandas.Dataframe constructor
        :param data: A frameable set of observations to assimilate into the distribution

        Iterates through every row in the dataframe and updates the mean/covariance
        with the values in that row.
        """
        # transorm dataframe into ndarray with properly ordered columns
        if isinstance(data, pd.DataFrame):
            self._validate_frame(data)
            data = data[self.labels].values

        if data.shape[1] != len(self._mu):
            raise ValueError('JointNormal tried to ingest data with wrong number of dimensions')

        for ind in range(data.shape[0]):
            x = self._vectorize(data[ind, :])
            self._ingest_point(x)

    def _prob_for_records(self, data, variable, op):
        """
        :type data: iterable
        :param data: An iterable of variables at which to evaluate threshold probability

        :type variable: str
        :param variable: the name of the thresholding variable

        :type op: str
        :param op: one of ['__lt', '__gt']

        :rtype: scalar or iterable depending on data
        :return: the output probability value(s)
        """
        # extract the single non-marginalized variable from the data
        x = data

        # get a distribution that marginalizes out all other variables
        p = self.marginal(variable)

        # get normal params that are now scalers since we've reduced to 1-d
        mu = p._mu[0, 0]
        sigma = np.sqrt(float(p.cov[0, 0]))

        # find probability of x being less than data value using the z-score
        z = (x - mu) / sigma
        prob_less_than_x = 0.5 * (1 + erf(z / np.sqrt(2)))

        prob_for_op = {
            '__lt': lambda x: list(prob_less_than_x),
            '__gt': lambda x: list(1. - prob_less_than_x)
        }
        out = prob_for_op[op](x)
        return out if len(out) > 1 else out[0]

    def _log_density_for_records(self, data):
        """
        :type data: pandas.Dataframe
        :param data: Dataframe with rows of observations

        :rtype: scalar or iterable depending on data
        :return: the output probability value(s)
        """
        # make sure input is an nd-array with rows holding records
        x_matrix = data[self.labels].values

        # subtract mu from every record (note result is transposed)
        x_matrix = np.matrix(x_matrix.T - np.array(self._mu))

        # hit each row (record) with precision matrix
        y_matrix = self.cov.getI() * x_matrix

        # compute the exponential density argument for each record
        exp_argument = -0.5 * np.sum(np.array(x_matrix) * np.array(y_matrix), axis=0)

        # compute the log of the density prefactor
        # k: dimension of distribution
        # det: determinant of covariance matrix
        # log_prefactor: log of the multivariate normal normalization constant
        k = self._mu.shape[0]
        det = np.abs(np.linalg.det(self.cov))
        log_prefactor = np.log((2. * np.pi) ** -(k / 2.) / np.sqrt(det))

        # compute and return log probability
        out = list(log_prefactor + exp_argument)
        return out[0] if len(out) == 1 else out

    def marginal(self, *labels):
        """
        :type labels: string arguments
        :param labels: marginalize out all variables not passed in these string arguments

        :rtype: JointNormal
        :return: A joint normal distribution marginalized to include only specified variables

        Example:

        .. code-block:: python

            N = JointNormal(
                labels=['x', 'y', 'z'],
                mu=[0, 0, 0],
                cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )
            # marginalize out the z variable
            N_cond = N.marginal('x', 'y')
        """
        # get the index of the labels you don't want to marginalize over
        ind = self._index_for(labels)

        # create a list of all combinations indexes you want to keep
        in_els = itertools.product(ind, ind)

        # create list of all combinations of row, columns indexes for the output covariance matrix
        out_els = itertools.product(range(len(ind)), range(len(ind)))

        # initialize an output covariance matrix
        cov_out = np.matrix(np.zeros((len(ind), len(ind))))

        # map appropriate input covariance matrix elements to output covariance matrix
        for in_el, out_el in zip(in_els, out_els):
            cov_out[out_el] = self.cov[in_el]

        # extract the mean elements into the output mean vector
        mu_out = self._mu[ind, 0]

        return JointNormal(mu=mu_out, cov=cov_out, N=self.N, labels=self._labels_for(ind), n_max=None)

    def _check_args(self, free_ind=None, fixed_ind=None):
        free_ind = [] if free_ind is None else free_ind
        fixed_ind = [] if fixed_ind is None else fixed_ind
        # k is dimensionality of space
        k = self._mu.shape[0]

        # make sure all indices are within dimensionality
        if any([ind >= k for ind in free_ind + fixed_ind]):
            raise ValueError('All indices must be less than {}'.format(k))

        # make sure there are no overlaps
        if len(set(free_ind).intersection(set(fixed_ind))) > 0:
            raise ValueError('An index cannot appear in both free_ind and fixed_ind')

        # make sure no dups
        if len(set(free_ind)) != len(free_ind) or len(set(fixed_ind)) != len(fixed_ind):
            raise ValueError('free_ind and fixed_ind cannot contain duplicate entries')

    def conditional(self, **constraints):
        """
        :type constraints: numbers
        :param constraints: constraints expressed as `variable_name=variable_value`

        :rtype: JointNormal
        :return: A joint normal distribution conditioned on supplied constraints

        Example:

        .. code-block:: python

            N = JointNormal(
                labels=['x', 'y', 'z'],
                mu=[0, 0, 0],
                cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )
            # Returns a distribution over z
            N_cond = N.conditional(x=1, y=1)
        """
        # separate constraints into index and values
        fixed_labels, fixed_vals = zip(*constraints.items())
        fixed_ind = self._index_for(fixed_labels)

        # the original index is just all integers up to dimension
        ind_orig = range(self._mu.shape[0])

        # get all indices that haven't been specified as fixed
        non_fixed_ind = list(set(ind_orig) - set(fixed_ind))

        # check indices for errors
        self._check_args(non_fixed_ind, fixed_ind)

        # permute covariance and mu to have non-fixed elements first, then fixed
        ind_perm = list(non_fixed_ind) + list(fixed_ind)
        P = _compute_permutation_matrix(ind_orig, ind_perm)
        mu_perm = P * self._mu
        cov_perm = P * self.cov * P.T

        # Follow the notation from the wikipedia multivariate normal article to partition the
        # covariance matrix into fixed and non-fixed parts.
        #
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        #
        # N1: dimension of non-fixed covariance partition
        # mu1: non-fixed mean vector
        # mu2: fixed mean vector
        # S11: fixed covariance matrix
        # S22: non-fixed covariance matrix
        # S12: covariance betwen fixed and not fixed variables
        # S21: transpose of S12
        # a: vector of fixed values on which the distribution is conditioned
        # mu: mean vector of conditioned distrubtion
        # S: covariance matrix of condidtioned distribution
        N1 = len(non_fixed_ind)
        mu1 = mu_perm[:N1, 0]
        mu2 = mu_perm[N1:, 0]
        S11 = cov_perm[:N1, :N1]
        S22 = cov_perm[N1:, N1:]
        S12 = cov_perm[:N1, N1:]
        S21 = cov_perm[N1:, :N1]
        a = np.matrix([[x] for x in fixed_vals])

        # formula for conditional distribution of partitioned normal distribution
        mu = mu1 + S12 * S22.getI() * (a - mu2)
        S = S11 - S12 * S22.getI() * S21

        # create a joint distrubtion out of the non-fixed params
        return JointNormal(mu=mu, cov=S, N=self.N, labels=self._labels_for(non_fixed_ind), n_max=None)

    @property
    def _info(self):
        return 'N({})'.format(','.join([str(s) for s in self.labels]))

    def __mul__(self, other):
        return multiply_independent_normals(self, other)

    def __add__(self, other):
        return add_independent_normals(self, other)

    def __sub__(self, other):
        other = copy.deepcopy(other)
        other._mu = - other._mu
        return add_independent_normals(self, other)

    def __gt__(self, other):
        return probability_first_gt_second(self, other)

    def __lt__(self, other):
        return probability_first_gt_second(other, self)

    def __repr__(self):
        return self._info

    def __str__(self):
        return self._info

    def estimate(self, variable, **constraints):
        """
        :type variable: str
        :param variable: The name of the variable to estimate

        :type constraints: numbers
        :param constraints: constraints expressed as `variable_name=variable_value`

        :rtype: tuple
        :return: (mean, standard_deviation)

        Returns a tuple of (mu, sigma) representing the mean and standard deviation
        of a particular variable given optional constraints for conditioning.  This
        method is useful for getting estimates of a particular variable.
        """
        # do any requested conditioning
        if constraints:
            out_normal = self.conditional(**constraints)
        else:
            out_normal = self

        out_normal = out_normal.marginal(variable)
        return out_normal._mu[0, 0], np.sqrt(out_normal.cov[0, 0])

    def percentile(self, **values):
        """
        :type values: numbers
        :param values: percentiles at which to compute locations.  Must be in the form
                       `variable_name=variable_value`.  variable_value can either be
                       a number or an iterable of numbers

        :rtype: scalar or iterable depending on input
        :return: the values corresponding to the input percentiles

        Example:

        .. code-block:: python

            N = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1, 0], [0, 1]])

            # compute the x value at the 80th percentile
            prob = N.percentile(x=.8)

            # compute x values for quartiles
            prob = N.percentile(x=[.25, .5, .75])
        """
        if len(values) > 1:
            raise ValueError('You can only compute percentiles for one variable')
        variable = list(values.keys())[0]
        data = values[variable]

        marginalized = self.marginal(variable)
        out = stats.norm(marginalized.mu[0, 0], np.sqrt(marginalized.cov[0, 0])).ppf(data)
        if hasattr(out, '__iter__'):
            return list(out)
        else:
            return out

    def probability(self, **constraints):
        """
        :type constraints: numbers
        :param constraints: constraints expressed as `variable_name=variable_value`.
                            One of the constraints must be expressed as either
                            `variable_name__gt=value` or `variable_name__lt=value`.
                            This constraint specifies the threshold value for
                            computing probabilities.  This contstraint can be either
                            a number or an iterable.

        :rtype: scalar or iterable depending on threshold constraint
        :return: the output probability value(s)


        Example:

        .. code-block:: python

            N = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1, 1], [1, 2]])

            # compute probability that y < 2 given that x = 1.
            prob = N.probability(y__lt=2, x=1)

            # compute probability that y < w for w in range(3) given that x = 1
            probs = N.probability(y__lt=range(3), x=1)

        """
        free_keys = [k for k in constraints.keys() if '__gt' in k or '__lt' in k]
        if len(free_keys) != 1:
            raise ValueError('You must supply at least one kwarg ending in __gt or __lt')
        free_key = free_keys[0]
        conditional_keys = set(constraints.keys()) - set(free_keys)
        out_norm = self
        if conditional_keys:
            out_norm = out_norm.conditional(**{k: constraints[k] for k in conditional_keys})

        variable = free_key[:-4]
        op = free_key[-4:]
        data = constraints[free_key]
        data = data if hasattr(data, '__iter__') else [data]
        return out_norm._prob_for_records(data, variable, op)

    def log_density(self, data, **constraints):
        """
        :type data: a dict or a valid argument to pandas.Dataframe constructor
        :param data: location(s) at which to compute log_density

        :type constraints: numbers
        :param constraints: constraints expressed as `variable_name=variable_value`.

        :rtype: scalar or iterable depending on threshold constraint
        :return: log of the output probability density value(s)

        First conditions on the constraints, then evaluates the remaining distribution
        at the location specified by data.  Any variable not explicitly included in
        either data or constrains is marginalized out.
        """
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

        # do any requested conditioning
        if constraints:
            out_norm = self.conditional(**constraints)
        else:
            out_norm = self

        # marginalize to selected variables
        out_norm = out_norm.marginal(*list(df.columns))

        # return the log densities corresponding to each frame record
        return out_norm._log_density_for_records(df)


def add_independent_normals(*distributions):
    """
    :type distributions: JointNormal
    :param distributions: a variable set of JointNormal instances (assumed to be independent)

    :rtype: JointNormal
    :return: The distribution obtained by multiplying together all input distributions.
    """
    normals = list(distributions[0]) if hasattr(distributions[0], '__iter__') else distributions

    labels = normals[0].labels
    labels_set = set(labels)

    for norm in normals:
        if set(norm.labels) != labels_set:
            raise ValueError('Cannot add distributions with different labels')

    cov_out = sum([norm.cov for norm in normals])
    mean_out = sum([norm._mu for norm in normals])

    # set N to None so that you can't incrementally add to this without explicitly setting N
    return JointNormal(labels=labels, mu=mean_out, cov=cov_out, N=None, n_max=None)


def multiply_independent_normals(*distributions):
    """
    :type distributions: JointNormal
    :param distributions: a variable set of JointNormal instances (assumed to be independent)

    :rtype: JointNormal
    :return: The distribution obtained by multiplying together all input distributions.
    """
    normals = list(distributions[0]) if hasattr(distributions[0], '__iter__') else distributions
    means_in = [norm._mu for norm in normals]
    precisions_in = [norm.cov.getI() for norm in normals]
    weighted_mean = sum([prec * mu for (prec, mu) in zip(precisions_in, means_in)])
    cov_out = sum(precisions_in).getI()
    mean_out = cov_out * weighted_mean
    labels = normals[0].labels
    labels_set = set(labels)
    for norm in normals:
        if set(norm.labels) != labels_set:
            raise ValueError('Cannot multiply distributions with different labels')

    # set N to None so that you can't incrementally add to this without explicitly setting N
    return JointNormal(labels=labels, mu=mean_out, cov=cov_out, N=None, n_max=None)


def probability_first_gt_second(first_norm, second_norm):
    """
    :type data: JointNormal
    :param first_norm: The first normal to compare (must be 1 dimentional)

    :type data: JointNormal
    :param second_norm: The second normal to compare (must be 1 dimentional)

    :rtype: float
    :return: probability that a draw from first distribution will be greater than a draw from second
    """
    if first_norm.mu.shape[0] != 1 or second_norm.mu.shape[0] != 1:
        raise ValueError('One of the distributions is not 1 dimensional.  Try marginalizing.')

    # labels don't need to be the same for this calculation since there is only one variable involved
    first_norm = copy.deepcopy(first_norm)
    first_norm.labels = ['x']
    second_norm = copy.deepcopy(second_norm)
    second_norm.labels = ['x']

    diff_norm = first_norm - second_norm
    mu = diff_norm._mu[0, 0]
    sigma = np.sqrt(float(diff_norm.cov[0, 0]))
    z = mu / sigma
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def _compute_permutation_matrix(initial_index, final_index):
    """
    Compute the permutation matrix that takes initial_index to final_index
    initial_index: an iterable of indices for the initial unpermuted elements
                   (must contain all integers in range(len(initial_index))
    final_index: an iterable of indices for the initial permuted elements
                   (must contain all integers in range(len(final_index))
    """
    permutation_matrix = np.matrix(np.zeros((len(initial_index), len(initial_index))))
    for final, initial in zip(final_index, initial_index):
        permutation_matrix[initial, final] = 1
    return permutation_matrix
