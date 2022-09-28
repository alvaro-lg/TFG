import math
import numpy as np

import math
import numpy as np

# Constates relativas a la compresión con la mu-law
N_BITS = 8
MU = 256
AMPLITUDES = (-1, 1)

class Mu_law_encoder:

    def __init__(self, mu=MU, n_bits=N_BITS, amps=AMPLITUDES):
        self.mu = mu
        self.n_levels = (2 ** n_bits) - 1
        self.min_amp, self.max_amp = amps

    def encode_sample(self, sample):

        # Transformation
        amp = np.sign(sample) * (np.log(1 + self.mu * np.abs(sample / self.max_amp))) / np.log(self.mu + 1)

        # Quantization
        return np.rint((self.n_levels * (amp-self.min_amp)) / (self.max_amp - self.min_amp))

    def decode_sample(self, idx):

        # De-Quantization
        amp = ((self.max_amp - self.min_amp) * idx / self.n_levels) + self.min_amp

        # De-Transforming
        return np.sign(amp) * self.max_amp * (((1 + self.mu) ** (np.abs(amp))) - 1) / self.mu

    def encode_series(self, data):

        return np.vectorize(self.encode_sample)(data)

    def decode_series(self, data):

        return np.vectorize(self.decode_sample)(data)