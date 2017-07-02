from unittest import TestCase
import numpy as np
import pandas as pd
from scipy import stats

from norma.joint_normal import (
    JointNormal,
    multiply_independent_normals,
    add_independent_normals,
    _compute_permutation_matrix
)


class BaseTest(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'x': [.2 * (n % 5) + (n % 4) for n in range(20)],
                'y': [float(n) for n in range(20)],
            }
        )
        self.cov = np.cov(self.df.x, self.df.y, ddof=0)

        self.mu = self.df.mean().values
        self.N = len(self.df)


class NonInitializedTests(BaseTest):
    def test_bad_mu_size(self):
        with self.assertRaises(ValueError):
            JointNormal(labels=['x', 'y'], mu=[1, 2, 3])

    def test_bad_cov_size(self):
        with self.assertRaises(ValueError):
            JointNormal(labels=['x', 'y'], cov=[[1]])

    def test_mean(self):
        p = JointNormal(labels=['x', 'y'])
        p.ingest(self.df)
        self.assertAlmostEqual(p.mu[0, 0], self.mu[0])
        self.assertAlmostEqual(p.mu[1, 0], self.mu[1])
        self.assertAlmostEqual(self.N, p.N)

    def test_numpy_input(self):
        p = JointNormal(labels=['x', 'y'])
        p.ingest(self.df.values)
        self.assertAlmostEqual(p.mu[0, 0], self.mu[0])
        self.assertAlmostEqual(p.mu[1, 0], self.mu[1])
        self.assertAlmostEqual(self.N, p.N)

    def test_numpy_bad_input(self):
        p = JointNormal(labels=['x', 'y'])
        self.df['z'] = self.df.x
        with self.assertRaises(ValueError):
            p.ingest(self.df.values)

    def test_n_limit(self):
        p = JointNormal(labels=['x', 'y'], n_max=5)
        p.ingest(self.df.values)
        self.assertAlmostEqual(5, p.N)

    def test_cov(self):
        p = JointNormal(labels=['x', 'y'])
        p.ingest(self.df)
        self.assertAlmostEqual(p.cov[0, 0], self.cov[0, 0])
        self.assertAlmostEqual(p.cov[1, 1], self.cov[1, 1])
        self.assertAlmostEqual(p.cov[1, 0], self.cov[1, 0])
        self.assertAlmostEqual(p.cov[0, 1], self.cov[0, 1])
        self.assertAlmostEqual(self.N, p.N)


class InitializedTests(BaseTest):
    def test_list_input_bad_frame(self):
        p = JointNormal(labels=['a', 'b'], mu=[0, 0], cov=[[1e-19, 1e-19], [1e-19, 1e-19]])
        with self.assertRaises(ValueError):
            p.ingest(self.df)

    def test_list_input(self):
        p = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1e-19, 1e-19], [1e-19, 1e-19]])
        df = self.df[['y', 'x']]
        p.ingest(df)
        mu, cov, N = p.mu_cov_n
        self.assertAlmostEqual(p.mu[0, 0], self.mu[0])
        self.assertAlmostEqual(p.mu[1, 0], self.mu[1])
        self.assertEqual(tuple(cov.shape), (2, 2))
        self.assertEqual(N, len(self.df))

    def test_array_input(self):
        p = JointNormal(labels=['x', 'y'], mu=np.array([0, 0]), cov=np.array([[1e-19, 1e-19], [1e-19, 1e-19]]))
        p.ingest(self.df)
        self.assertAlmostEqual(p.mu[0, 0], self.mu[0])
        self.assertAlmostEqual(p.mu[1, 0], self.mu[1])

    def test_set_mu(self):
        p = JointNormal(labels=['x', 'y'], mu=np.array([0, 0]), cov=np.array([[1e-19, 1e-19], [1e-19, 1e-19]]))
        p.mu = self.mu
        self.assertAlmostEqual(p.mu[0, 0], self.mu[0])
        self.assertAlmostEqual(p.mu[1, 0], self.mu[1])

    def test_set_cov(self):
        p = JointNormal(labels=['x', 'y'], mu=np.array([0, 0]), cov=np.array([[1e-19, 1e-19], [1e-19, 1e-19]]))
        p.cov = self.cov
        self.assertAlmostEqual(p.cov[0, 0], self.cov[0, 0])
        self.assertAlmostEqual(p.cov[1, 1], self.cov[1, 1])

    def test_set_mu_bad(self):
        p = JointNormal(labels=['x', 'y'], mu=np.array([0, 0]), cov=np.array([[1e-19, 1e-19], [1e-19, 1e-19]]))
        with self.assertRaises(ValueError):
            p.mu = [1, 2, 3]

    def test_set_cov_bad(self):
        p = JointNormal(labels=['x', 'y'], mu=np.array([0, 0]), cov=np.array([[1e-19, 1e-19], [1e-19, 1e-19]]))
        with self.assertRaises(ValueError):
            p.cov = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


class InfoTests(BaseTest):
    def test_repr(self):
        p = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1e-19, 1e-19], [1e-19, 1e-19]])
        p.ingest(self.df)
        s = repr(p)
        self.assertTrue('N' in s)
        self.assertTrue('x' in s)
        self.assertTrue('x' in s)

    def test_str(self):
        p = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[1e-19, 1e-19], [1e-19, 1e-19]])
        p.ingest(self.df)
        s = str(p)
        self.assertTrue('N' in s)
        self.assertTrue('x' in s)
        self.assertTrue('x' in s)


class RenameMeTests(TestCase):
    def setUp(self):
        self.P = JointNormal(
            mu=[1, 1],
            labels=['x', 'y'],
            cov=[
                [1, 1],
                [1, 1.01],
            ], N=10)

    def test_index_for_bad_labels(self):
        with self.assertRaises(ValueError):
            self.P._index_for(['bad', 'labels'])

    def test_mu_frame(self):
        df = self.P.mu_frame
        self.assertEqual(list(df.columns), ['mu'])
        self.assertEqual(len(df), 2)

    def test_cov_frame(self):
        df = self.P.cov_frame
        self.assertEqual(list(df.columns), ['x', 'y'])
        self.assertEqual(list(df.index), ['x', 'y'])

    def test_check_args_too_big(self):
        with self.assertRaises(ValueError):
            self.P._check_args(free_ind=[100])

    def test_check_args_overlap(self):
        with self.assertRaises(ValueError):
            self.P._check_args(free_ind=[1], fixed_ind=[1])

    def test_check_args_dups_free(self):
        with self.assertRaises(ValueError):
            self.P._check_args(free_ind=[1, 1])

    def test_check_args_dups_fixed(self):
        with self.assertRaises(ValueError):
            self.P._check_args(fixed_ind=[1, 1])


class EstimateTests(TestCase):
    def test_marginal(self):
        P = JointNormal(
            mu=[-1, 1],
            labels=['x', 'y'],
            cov=[
                [1, 1],
                [1, 1.01],
            ], N=10)
        mu_x, sigma_x = P.estimate('x')
        mu_y, sigma_y = P.estimate('y')
        self.assertAlmostEqual(mu_x, -1)
        self.assertAlmostEqual(mu_y, 1)
        self.assertAlmostEqual(sigma_x, 1)
        self.assertAlmostEqual(sigma_y, 1, places=1)

    def test_conditional(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 1],
                [1, 1.01],
            ], N=10)
        mu_x, sigma_x = P.estimate('x', y=9)
        self.assertAlmostEqual(mu_x, 8.91, places=2)
        self.assertAlmostEqual(sigma_x, .1, places=1)


class ProbabilityTests(TestCase):
    def test_no_limits(self):
        P = JointNormal(
            mu=[1, 1],
            labels=['x', 'y'],
            cov=[
                [1, 1],
                [1, 1.01],
            ], N=10)
        with self.assertRaises(ValueError):
            P.probability(x=2, y=2)

    def test_sums_to_one(self):
        P = JointNormal(
            mu=[1, 1],
            labels=['x', 'y'],
            cov=[
                [1, 1],
                [1, 1.01],
            ], N=10)
        px_gt = P.probability(x__gt=2, y=2)
        px_lt = P.probability(x__lt=2, y=2)
        py_gt = P.probability(y__gt=2, x=-20)
        py_lt = P.probability(y__lt=2, x=-20)
        self.assertAlmostEqual(sum([px_gt, px_lt]), 1, places=14)
        self.assertAlmostEqual(sum([py_gt, py_lt]), 1, places=14)

    def test_correct_value(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 0],
                [0, 1],
            ], N=10)
        N = stats.norm()
        self.assertAlmostEqual(P.probability(x__gt=0, y=0), N.cdf(0), places=14)
        self.assertAlmostEqual(P.probability(x__lt=1, y=0), N.cdf(1), places=14)
        self.assertAlmostEqual(P.probability(x__gt=5, y=0), 1. - N.cdf(5), places=14)

    def test_iterable(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 0],
                [0, 1],
            ], N=10)
        x0 = range(-5, 6)
        p_gt = np.array(P.probability(x__gt=x0))
        p_lt = np.array(P.probability(x__lt=x0, y=2))
        p_sum = p_gt + p_lt
        self.assertTrue(all(round(x, 2) == 1 for x in p_sum))
        self.assertAlmostEqual(p_gt[5], .5)
        self.assertTrue(p_gt[0] > .5)
        self.assertTrue(p_lt[0] < .5)
        self.assertTrue(p_gt[-1] < .5)
        self.assertTrue(p_lt[-1] > .5)


class PercentileTests(TestCase):
    def test_too_many_vars(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 0],
                [0, 1],
            ], N=10)
        with self.assertRaises(ValueError):
            P.percentile(x=.75, y=.2)

    def test_correct_value(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 0],
                [0, 1],
            ], N=10)
        percentile_value = P.percentile(x=.75)
        self.assertAlmostEqual(P.probability(x__lt=percentile_value), .75, places=12)

    def test_iterable(self):
        P = JointNormal(
            mu=[0, 0],
            labels=['x', 'y'],
            cov=[
                [1, 0],
                [0, 1],
            ], N=10)
        percentiles = [.25, .5, .75]
        values_at_percentiles = P.percentile(x=percentiles)
        new_percentiles = P.probability(x__lt=values_at_percentiles)
        self.assertAlmostEqual(new_percentiles[0], percentiles[0])
        self.assertAlmostEqual(new_percentiles[1], percentiles[1])
        self.assertAlmostEqual(new_percentiles[2], percentiles[2])


class LogDensityTests(TestCase):
    def setUp(self):
        self.norm = JointNormal(
            mu=[0, 1, 2, 3],
            labels=['a', 'b', 'c', 'd'],
            cov=[
                [1, 2, 3, 4],
                [2, 5, 6, 7],
                [3, 6, 8, 9],
                [4, 7, 9, 10],
            ], N=10)

    def test_bayes(self):
        P = self.norm
        X1 = {'a': 4, 'b': 3}
        X2 = {'c': 1.5, 'd': 1.7}
        X = {}
        X.update(X1)
        X.update(X2)
        log_p_x1_given_x2 = P.log_density(X1, **X2)
        log_p_x2_given_x1 = P.log_density(X2, **X1)
        log_p_x1 = P.log_density(X1)
        log_p_x2 = P.log_density(X2)

        log_pa = log_p_x1_given_x2 + log_p_x2
        log_pb = log_p_x2_given_x1 + log_p_x1
        log_pc = P.log_density(X)
        self.assertAlmostEqual(log_pa, log_pb)
        self.assertAlmostEqual(log_pa, log_pc)

    def test_frame(self):
        P = self.norm
        df = pd.DataFrame(np.ones((4, 2)), columns=['a', 'b'])
        unconditional = P.log_density(df)
        conditional = P.log_density(df, c=1, d=1)
        self.assertEqual(len(conditional), len(unconditional))
        self.assertTrue(isinstance(unconditional, list))
        self.assertTrue(isinstance(conditional, list))


class ConditionalTests(TestCase):
    def setUp(self):
        # Let R be the square root covariance matrix (lower diag)
        # Then imagine a sample where measurement errors are
        # e = [e_x, 0, 0].T
        # Then the conditional expectation of the joint normal
        # with any mu and covariance S = R * R.T
        # will be X_cond = mu + R * e

        # define sqrt covariance (this can be arbitrary square matrix)
        self.R = np.matrix(
            [
                [100, 0, 0],
                [20, 10, 0],
                [5, 4, 3]
            ]
        )

        # compute covariance matrix
        S = self.R * self.R.T

        # define the mu vector
        self.mu = np.matrix([2, 3, 4]).T

        # create joint normal
        self.N = JointNormal(mu=self.mu, labels=['x', 'y', 'z'], cov=S, N=10)

    def test_mu_condition_on_x(self):
        # define readability shortcuts
        N, R = self.N, self.R

        # set up errors defining conditioning on e_x = 1
        e = np.matrix([1, 0, 0]).T

        # set up expected conditional
        X = self.mu + R * e
        Xc = X[1:, 0]

        # condition the normal
        Nc = N.conditional(x=X[0, 0])

        # test that result is expected
        diffs = [Xc[nn, 0] - Nc.mu[nn, 0] for nn in range(Xc.shape[0])]
        self.assertTrue(all([np.abs(d) < 1e-6 for d in diffs]))

    def test_cov_condition_on_x(self):
        # define readability shortcuts
        N, R = self.N, self.R

        # set up errors defining conditioning on e_x = 1
        # since x is known, its variance (upper-left) goes to zero
        D = np.matrix(np.diag([0, 1, 1]))
        Sc = (R * D * R.T)[1:, 1:]

        # condition on any x value to get the same covariance
        Nc = N.conditional(x=17)
        diff = Sc - Nc.cov
        self.assertTrue(np.abs(diff.max()) < 1e-6)
        self.assertTrue(np.abs(diff.min()) < 1e-6)

    def test_cov_condition_on_xy(self):
        # define readability shortcuts
        N, R = self.N, self.R

        # set up errors defining conditioning on e_x = 1
        D = np.matrix(np.diag([0, 0, 1]))
        Sc = (R * D * R.T)[2:, 2:]

        # condition on any x value to get the same covariance
        Nc = N.conditional(x=17, y=23)
        diff = Sc - Nc.cov
        self.assertTrue(np.abs(diff.max()) < 1e-6)
        self.assertTrue(np.abs(diff.min()) < 1e-6)

    def test_mu_condition_on_xy(self):
        # define readability shortcuts
        N, R = self.N, self.R

        # set up errors defining conditioning on e_x = 1
        e = np.matrix([1, 1, 0]).T

        # set up expected conditional
        X = self.mu + R * e
        Xc = X[2:, 0]

        # condition the normal
        Nc = N.conditional(x=X[0, 0], y=X[1, 0])

        # test that result is expected
        diffs = [Xc[nn, 0] - Nc.mu[nn, 0] for nn in range(Xc.shape[0])]
        self.assertTrue(all([np.abs(d) < 1e-6 for d in diffs]))


class MarginalTests(TestCase):
    def setUp(self):
        self.norm = JointNormal(
            mu=[0, 1, 2],
            labels=['x', 'y', 'z'],
            cov=[
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]
            ], N=10)

    def test_forward(self):
        marg = self.norm.marginal('y', 'z')
        cov_diff = marg.cov - np.matrix([[4, 5], [7, 8]])
        mu_diff = marg.mu - np.matrix([[1], [2]])
        self.assertAlmostEqual(cov_diff.max(), 0)
        self.assertAlmostEqual(cov_diff.min(), 0)
        self.assertAlmostEqual(mu_diff.min(), 0)
        self.assertAlmostEqual(mu_diff.max(), 0)
        self.assertEqual(marg.N, self.norm.N)

    def test_backward(self):
        marg = self.norm.marginal('z', 'y')
        cov_diff = marg.cov - np.matrix([[8, 7], [5, 4]])
        mu_diff = marg.mu - np.matrix([[2], [1]])
        self.assertAlmostEqual(cov_diff.max(), 0)
        self.assertAlmostEqual(cov_diff.min(), 0)
        self.assertAlmostEqual(mu_diff.min(), 0)
        self.assertAlmostEqual(mu_diff.max(), 0)
        self.assertEqual(marg.N, self.norm.N)


class MultiplyTests(BaseTest):
    def test_diag_multiply(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 0], cov=[[2, 0], [0, 2]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[0, 1], cov=[[2, 0], [0, 2]])
        normp = norm1 * norm2

        V = normp.cov
        mu = normp.mu

        self.assertAlmostEqual(V[0, 0], 1)
        self.assertAlmostEqual(V[1, 1], 1)
        self.assertAlmostEqual(V[0, 1], 0)
        self.assertAlmostEqual(V[1, 0], 0)
        self.assertAlmostEqual(mu[0, 0], .5)
        self.assertAlmostEqual(mu[1, 0], .5)

    def test_full_multiply(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[2, 1], [1, 2]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[2, 1], [1, 2]])
        normp = norm1 * norm2

        prec1 = norm1.cov.getI()
        prec2 = norm2.cov.getI()
        precp = normp.cov.getI()
        dprec = precp - prec1 - prec2

        self.assertAlmostEqual(dprec.max(), 0)
        self.assertAlmostEqual(dprec.min(), 0)
        self.assertAlmostEqual(normp.mu.max(), 1)
        self.assertAlmostEqual(normp.mu.min(), 1)

    def test_full_add(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[2, 1], [1, 2]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[2, 1], [1, 2]])
        normp = norm1 + norm2

        mu_diff = normp.mu - (norm1.mu + norm2.mu)
        cov_diff = normp.cov - (norm1.cov + norm2.cov)

        self.assertAlmostEqual(mu_diff.max(), 0)
        self.assertAlmostEqual(mu_diff.min(), 0)
        self.assertAlmostEqual(cov_diff.max(), 0)
        self.assertAlmostEqual(cov_diff.min(), 0)

    def test_full_subtract(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[2, 1], [1, 2]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[2, 1], [1, 2]])
        normp = norm1 - norm2

        mu_diff = normp.mu - (norm1.mu - norm2.mu)
        cov_diff = normp.cov - (norm1.cov + norm2.cov)

        self.assertAlmostEqual(mu_diff.max(), 0)
        self.assertAlmostEqual(mu_diff.min(), 0)
        self.assertAlmostEqual(cov_diff.max(), 0)
        self.assertAlmostEqual(cov_diff.min(), 0)


class MultiplyNormalsTests(BaseTest):
    def test_with_bad_labels(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'z'], mu=[2, 2], cov=[[4, 1], [1, 4]])
        with self.assertRaises(ValueError):
            multiply_independent_normals(norm1, norm2)

    def test_with_args(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[4, 1], [1, 4]])
        normp = multiply_independent_normals(norm1, norm1, norm2, norm2)

        prec1 = norm1.cov.getI()
        prec2 = norm2.cov.getI()
        precp = normp.cov.getI()
        dprec = precp - 2 * (prec1 + prec2)

        self.assertAlmostEqual(dprec.max(), 0)
        self.assertAlmostEqual(dprec.min(), 0)
        self.assertAlmostEqual(normp.mu.max(), 1)
        self.assertAlmostEqual(normp.mu.min(), 1)

    def test_with_list(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[4, 1], [1, 4]])
        normp = multiply_independent_normals([norm1, norm1, norm2, norm2])

        prec1 = norm1.cov.getI()
        prec2 = norm2.cov.getI()
        precp = normp.cov.getI()
        dprec = precp - 2 * (prec1 + prec2)

        self.assertAlmostEqual(dprec.max(), 0)
        self.assertAlmostEqual(dprec.min(), 0)
        self.assertAlmostEqual(normp.mu.max(), 1)
        self.assertAlmostEqual(normp.mu.min(), 1)
        self.assertTrue(normp.N is None)


class AddNormalsTests(BaseTest):
    def test_with_bad_labels(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[0, 0], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'z'], mu=[2, 2], cov=[[4, 1], [1, 4]])
        with self.assertRaises(ValueError):
            add_independent_normals(norm1, norm2)

    def test_with_args(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[40, 1], [1, 40]])
        normp = add_independent_normals(norm1, norm1, norm2, norm2)

        cov1 = norm1.cov
        cov2 = norm2.cov
        covp = normp.cov
        dcov = covp - 2 * (cov1 + cov2)

        self.assertAlmostEqual(dcov.max(), 0)
        self.assertAlmostEqual(dcov.min(), 0)
        self.assertAlmostEqual(normp.mu.max(), 6)
        self.assertAlmostEqual(normp.mu.min(), 6)

    def test_with_list(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['x', 'y'], mu=[2, 2], cov=[[40, 1], [1, 4]])
        normp = add_independent_normals([norm1, norm1, norm2, norm2])

        cov1 = norm1.cov
        cov2 = norm2.cov
        covp = normp.cov
        dcov = covp - 2 * (cov1 + cov2)

        self.assertAlmostEqual(dcov.max(), 0)
        self.assertAlmostEqual(dcov.min(), 0)
        self.assertAlmostEqual(normp.mu.max(), 6)
        self.assertAlmostEqual(normp.mu.min(), 6)


class ProbabilityFirstGTSecondTests(BaseTest):
    def test_5050(self):
        norm1 = JointNormal(labels=['x'], mu=[1], cov=[[4]])
        norm2 = JointNormal(labels=['y'], mu=[1], cov=[[4]])
        self.assertAlmostEqual(norm1 > norm2, .5)
        self.assertAlmostEqual(norm2 > norm1, .5)

    def test_one_sigma_difference(self):
        norm1 = JointNormal(labels=['x'], mu=[2], cov=[[2]])
        norm2 = JointNormal(labels=['y'], mu=[0], cov=[[2]])
        self.assertAlmostEqual(norm1 > norm2, 0.84134474606854293)
        self.assertAlmostEqual(norm1 < norm2, 0.15865525393145707)

    def test_bad_dimensions(self):
        norm1 = JointNormal(labels=['x', 'y'], mu=[1, 1], cov=[[4, 1], [1, 4]])
        norm2 = JointNormal(labels=['y'], mu=[0], cov=[[2]])
        with self.assertRaises(ValueError):
            norm1 > norm2


class ComputePermutationTests(TestCase):
    def test_squaring(self):
        N = 6
        ind = range(N)
        ind_p = [ind[nn] for nn in range(N - 1, -1, -1)]
        P = _compute_permutation_matrix(ind, ind_p)
        P2 = P * P
        R = P2 - np.matrix(np.identity(N))
        self.assertAlmostEqual(R.max(), 0)
        self.assertAlmostEqual(R.min(), 0)
