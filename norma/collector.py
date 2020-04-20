


class Collector:
    def __init__(self, robust=False, covariance_type='biased'):
        self.allowed_covariance_types = ['biased', 'frequency', 'precision?']

    def ingest(self, data, weights=None, labels=None):
        """
        data: any object that can be passed to dataframe constructor
        labels: optional list of names for variables.
        """

    def set_prior_mean_dist(self, prior_dist):
        pass

    def get_sample_dist(self):
        pass

    def get_mean_dist(self):
        pass




