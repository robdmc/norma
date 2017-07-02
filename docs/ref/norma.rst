.. _ref-norma:

Code documentation
==================

Joint Normal Distributions
--------------------------
An instance of the `JointNormal` class is maintained for each grid point of a JointProbLearner that has defined normal
variables.  This class provides an interface for training and extracting information from joint-normal distributions.
There is quite a bit of math going on behind the scenes in this class.  A terse writeup providing the necessary
background for understanding it can be obtained from https://github.com/robdmc/stats_for_software .

Example:

.. code-block:: python

    from norma import JointNormal

    # Create a standard normal distribution assuming it was generated from 10 observations.
    # Set up to forget old observations using an n_max of 100 observations. 
    N = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1, 0], [0, 1]], N=10, n_max=100)

    # Add observations to the distribution
    N.ingest([
        {'x': 1.0, 'y': 3.0},
        {'x': -0.6, 'y': 1.2},
        {'x': -2.0, 'y': -4.3},
    ])

    # find marginal estimates for parameters
    x_mu, x_std = N.estimate('x')
    y_mu, y_std = N.estimate('y')

    # find conditional estimates for parameters
    x_mu, x_std = N.estimate('x', y=1)
    y_mu, y_std = N.estimate('y', x=1)

    # get log probability density at a point
    dens = N.log_density(dict(x=2, y=-1))

    # find probabilities that a variable exceeds some limit
    p_marginal = N.probability(x__gt=2)

    # find probabilities that y is less than threshold given a value for x
    p_conditional = N.probability(y__lt=2, x=1)

    # multiply two independent joint normal distributions together and look at params
    N1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1, 0], [0, 1]])
    N2 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[2, 0], [0, 2]])
    N3 = N1 * N2
    mu_x_3, sig_x_3 = N3.estimate('x')

    # create a 3-d distribution to demonstrate marginalization and conditioning
    N4 = JointNormal(
        labels=['x', 'y', 'z'],
        mu=[0, 0, 0],
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    # marginalize out the z variable
    N_marg = N4.marginalize('x', 'y')
    N_marg.labels  # will be equal to ['x', 'y']

    # condition on x and y variables
    N_cond = N4.conditional(x=1, y=1)
    N_cond.labels  # will be equal to ['z']




.. automodule:: norma.joint_normal
    :members:
