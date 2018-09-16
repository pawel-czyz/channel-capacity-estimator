==========================
Channel Capacity Estimator
==========================

Channel Capacity Estimator (**cce**) is a python module to estimate 
`information capacity`_ of a communication channel. Mutual information,
computed as proposed by `Kraskov et al.`_ (*Physical Review E*, 2004,
Eq. (8)), is maximized over input probabilities by means of a constrained 
gradient-based stochastic optimization. The only parameter of the Kraskov 
algorithm is the number of neighbors, *k*, used in the nearest neighbor 
search. In **cce**, channel input is expected to be of categorical type 
(meaning that it should be described by labels), whereas channel output
is assumed to be in the form of points in real space of any dimensionality. 

The code performs local gradient-based optimization that, owing to the
fact that mutual information is a concave function of input probabilities,
is able to locate global maximum of mutual information. Maximization is
performed according to ADAM algorithm as implemented in TensorFlow_.
To use **cce**, you should have TensorFlow (with python bindings) installed
on your system. See file requirements.txt for a complete list of dependencies.

Module **cce** features the research article "Limits to the rate of 
information transmission through MAPK pathway" by Grabowski *et al.*, 
(submitted 2018). Version 1.0 of **cce** (with pre-built documentation)
has been included as supplementary code of this article.

For any updates and fixes to **cce**, please visit project homepage:
http://pmbm.ippt.pan.pl/software/cce 
(this permalink currently directs to a GitHub repository:
https://github.com/pawel-czyz/channel-capacity-estimator).


Usage
-----

There are three major use cases of **cce**:

1. Calculation of mutual information (for equiprobable input distributions).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the example below, mutual information is calculated between three sets 
of points drawn at random from two-dimensional Gaussian distributions,
located at (0,0), (0,1), and at (3,3) (in SciPy, covariance matrices of 
all three distributions  by default are identity matrices). Auxiliary 
function `label_all_with` helps to prepare the list of all points, in 
which each point is labeled according to its distribution of origin.

.. code:: python

    >>> from scipy.stats import multivariate_normal as mvn
    >>> from cce import WeightedKraskovEstimator as wke
    >>>
    >>> def label_all_with(label, values): return [(label, v) for v in values]
    >>>
    >>> data = label_all_with('A', mvn(mean=(0,0)).rvs(10000)) \
             + label_all_with('B', mvn(mean=(0,1)).rvs(10000)) \
             + label_all_with('C', mvn(mean=(3,3)).rvs(10000))
    >>>
    >>> wke(data).calculate_mi(k=10)
    0.9552107248613955

In this example, probabilities of input distributions, henceforth referred
to as *weights*, are assumed to be equal for all input distributions. Format
of data is akin to [('A', array([-0.4, 2.8])), ('A', array([-0.9, -0.1])), 
..., ('B', array([1.7, 0.9])), ..., ('C', array([3.2, 3.3])), ...).
Entries of data are not required to be grouped according to the label.
Distribution labels can be given as strings, not just single characters. 
Instead of NumPy arrays, ordinary lists with coordinates will be also 
accepted. (This example involves random numbers, so your result may vary
slightly.)


2. Calculation of mutual information for input distributions with non-equal probabilities.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example is structured as above, with an addition of weights of each 
input distributions:

.. code:: python

    >>> from scipy.stats import multivariate_normal as mvn
    >>> from cce import WeightedKraskovEstimator as wke
    >>>
    >>> def label_all_with(label, values): return [(label, v) for v in values]
    >>>
    >>> data = label_all_with('A', mvn(mean=(0,0)).rvs(10000)) \
             + label_all_with('B', mvn(mean=(0,1)).rvs(10000)) \
             + label_all_with('C', mvn(mean=(3,3)).rvs(10000))
    >>>
    >>> weights = {'A': 2/6, 'B': 1/6, 'C': 3/6}
    >>> wke(data).calculate_weighted_mi(weights=weights, k=10)
    1.0065891280377155

(This example involves random numbers, so your result may vary slightly.)


3. Estimation of channel capacity by maximizing MI with respect to input weights.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> from scipy.stats import multivariate_normal as mvn
    >>> from cce import WeightedKraskovEstimator as wke
    >>>
    >>> def label_all_with(label, values): return [(label, v) for v in values]
    >>>
    >>> data = label_all_with('A', mvn(mean=(0,0)).rvs(10000)) \
             + label_all_with('B', mvn(mean=(0,1)).rvs(10000)) \
             + label_all_with('C', mvn(mean=(3,3)).rvs(10000))
    >>>
    >>> wke(data).calculate_maximized_mi(k=10)
    (1.0154510500713743, {'A': 0.33343804, 'B': 0.19158363, 'C': 0.4749783})

The output tuple contains the maximized mutual information (channel capacity) 
and probabilities of input distributions that maximize mutual information (argmax). 
Optimization is performed within TensorFlow with multiple threads and takes 
less than a minute on a quad-core processor.
(This example involves random numbers, so your result may vary slightly.)


Testing
-------
To launch a suite of unit tests, run:

.. code:: bash

    $ make test


Documentation
-------------
Developer's code documentation may be generated with:

.. code:: bash

   $ cd docs
   $ make html


Installation
------------
To install **cce** locally via pip, run:

.. code:: bash

    $ make install

Then, you can directly start using the package:

.. code:: bash

    $ python
    >>> from cce import WeightedKraskovEstimator
    >>> ...


Authors
-------

The code was developed by `Frederic Grabowski`_ and `Paweł Czyż`_,
with some guidance from `Marek Kochańczyk`_ and under supervision of 
`Tomasz Lipniacki`_ from the `Laboratory of Modeling in Biology and Medicine`_,
`Institute of Fundamental Technological Reasearch, Polish Academy of Sciences`_
(IPPT PAN) in Warsaw.


License
-------

This software is distributed under `GNU GPL 3.0 license`_.


.. _information capacity: https://en.wikipedia.org/wiki/Channel_capacity
.. _Kraskov et al.: https://doi.org/10.1103/PhysRevE.69.066138
.. _TensorFlow:       https://www.tensorflow.org
.. _Frederic Grabowski: https://github.com/grfrederic
.. _Paweł Czyż: https://github.com/pawel-czyz
.. _Marek Kochańczyk: http://pmbm.ippt.pan.pl/web/Marek_Kochanczyk
.. _Tomasz Lipniacki: http://pmbm.ippt.pan.pl/web/Tomasz_Lipniacki
.. _Laboratory of Modeling in Biology and Medicine: http://pmbm.ippt.pan.pl
.. _Institute of Fundamental Technological Reasearch, Polish Academy of Sciences: http://www.ippt.pan.pl
.. _GNU GPL 3.0 license: https://www.gnu.org/licenses/gpl-3.0.html

