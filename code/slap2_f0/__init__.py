"""Blind-spot network that estimates the baseline (F0) and noise of SLAP2 glutamate movies.

Modules:
    io       read trial files written by matlab/exportF0TrainingData.m
    data     patch sampling for training
    model    blind-spot networks (the output at a pixel and frame never sees
             the input inside its blind spot)
    losses   heteroscedastic likelihoods for the residual target
"""

__version__ = "0.1.0"
