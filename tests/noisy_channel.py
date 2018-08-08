import numpy as np

def communicate(input_counts: dict, output_values: dict, sigma: float = 1e-4) -> list:
    """Creates a list of pairs (input, [outputs]).

    A noisy information channel can be treated as a device that converts
    non-deterministically an input into an output. The input are labels
    (that are keys of `input_counts`), whereas the output is a numerical
    value. Mutual information measures how much information about input
    (labels) we can retrieve just by observing output values (real values).
    This function generates outputs randomly around a given target value.

    Parameters
    ----------
    input_counts : dict
        dictionary of input labels (str or int) -> amounts of signals to
        be transmitted (int)

    output_values : dict
        dictionary of input labels (str or int) -> expected output of each
        input signal (float)

    sigma : float
        standard deviation controlling noisiness of the communication channel

    Returns
    -------
    list
        list of pairs describing generated samples. Example:
        [
            ('A', [0.001]),
            ('A', [-0.001]),
            ('B', [-0.002]),
            ...
        ]
    """
    accum = []
    for input_label, amount in input_counts.items():
        for _ in range(amount):
            outcomes = output_values[input_label] + sigma*np.random.normal()
            accum.append((input_label, [outcomes]))
    return accum

