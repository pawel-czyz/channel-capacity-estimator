==========================
Channel Capacity Estimator
==========================

The aim of this tool is to estimate information channel capacity.

Mutual information, computed as proposed by
Kraskov *et al.* [Crossref_, arXiv_],
is maximized over input probabilities
by means of a gradient-based stochastic optimization.

Usage
-----
This module can be used to:
 - standard MI calculation
 - MI calculation with given input (X) weights
 - estimation of channel capacity - that is finding maximal MI with input probabilities that maximalize it

.. code:: python

    >>> from cce import WeightedKraskovEstimator
    >>> data = [("label1", [1, 0, 2]), "label1": [1.1, 0.1, 2], ("label2", [1.1, 1, 5]), ...]
    >>> "Calculating usual MI"
    >>> wke = WeightedKraskovEstimator(data)
    >>> print(wke.calculate_mi(k=10))
    >>> 0.4
    >>> "Calculating MI with assumption that dataset is not balanced"
    >>> weights = {"label1": 1/2, "label2": 1/4, "label3": 1/4}
    >>> print(wke.calculate_weighted_mi(weights=weights, k=10))
    >>> "Finding weights that maximalise MI. This is exactly channel capacity."
    >>> print(wke.calculate_neighborhood(k=10))
    >>> print(wke.optimize_weights())
    >>> 0.7, {"label1": 0.6, "label2": 0.3, "label3": 0.1}

Installation
------------
To install just run

.. code:: bash

    $ make test

Then, you can directly start using the package:

.. code:: bash

    $ python
    >>> from cce import WeightedKraskovEstimator
    >>> ...

or run unit tests:

.. code:: bash

    $ make test



Authors
-------

`Laboratory of Modeling in Biology and Medicine`_
 - `Frederic Grabowski`_
 - `Paweł Czyż`_
 - `Marek Kochańczyk`_
 - `Tomasz Lipniacki`_

.. _arXiv:    https://arxiv.org/pdf/cond-mat/0305641.pdf
.. _CrossRef: https://doi.org/10.1103/PhysRevE.69.066138
.. _Frederic Grabowski: https://github.com/grfrederic
.. _Paweł Czyż: https://github.com/pawel-czyz
.. _Marek Kochańczyk: http://www.ippt.pan.pl/en/staff/mkochan
.. _Tomasz Lipniacki: http://www.ippt.pan.pl/en/staff/tlipnia
.. _Laboratory of Modeling in Biology and Medicine: http://pmbm.ippt.pan.pl/web/Main_Page
