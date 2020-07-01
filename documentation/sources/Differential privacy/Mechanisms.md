<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L8)</span>
### RandomizedResponseCoins

```python
shfl.differential_privacy.dp_mechanism.RandomizedResponseCoins(prob_head_first=0.5, prob_head_second=0.5)
```


This class uses a simple mechanism to add randomness for binary data. This algorithm is described
by Cynthia Dwork and Aaron Roth in "The algorithmic Foundations of Differential Privacy".

1.- Flip a coin

2.- If tails, then respond truthfully.

3.- If heads, then flip a second coin and respond "Yes" if heads and "No" if tails.

Input data must be binary, otherwise exception will be raised.

__Arguments__

- __prob_head_first__: float in [0,1] representing probability to use a random response instead of true value.
    This is equivalent to prob_head of the first coin flip algorithm described by Dwork.
- __prob_head_second__: float in [0,1] representing probability of respond true when random answer is provided.
    Equivalent to prob_head in the second coin flip in the algorithm.

__References__

- [The algorithmic foundations of differential privacy](
   https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L53)</span>
### RandomizedResponseBinary

```python
shfl.differential_privacy.dp_mechanism.RandomizedResponseBinary(f0, f1)
```


Implements the most general binary randomized response algorithm. Both the input and output are binary
arrays or scalars. The algorithm is defined through the conditional probabilities

- P( output=0 | input=0 ) = f0
- P( output=1 | input=1) = f1

For f0=f1=0 or 1, the algorithm is not random. It is maximally random for f0=f1=1/2.
This class contains, for special cases of f0, f1, the class RandomizedResponseCoins.

Input data must be binary, otherwise exception will be raised.

__Arguments__

- __f0__: float in [0,1] representing the probability of getting 0 when the input is 0
- __f1__: float in [0,1] representing the probability of getting 1 when the input is 1
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L100)</span>
### LaplaceMechanism

```python
shfl.differential_privacy.dp_mechanism.LaplaceMechanism(sensitivity, epsilon, query=None)
```


Implements the Laplace mechanism for differential privacy defined by Dwork in
"The algorithmic Foundations of Differential Privacy".

Notice that the Laplace mechanism is a randomization algorithm that depends on the sensitivity,
which can be regarded as a numeric query. One can show that this mechanism is
epsilon-differentially private with epsilon = sensitivity/b where b is a constant.

In order to apply this mechanism for a particular value of epsilon, we need to compute
the sensitivity, which might be hard to compute in practice. The framework provides
a method to estimate the sensitivity of a query that maps the private data in a normed space
(see: [SensitivitySampler](../Sensitivity Sampler))

__Arguments:__

- __sensitivity__: float representing sensitivity of the applied query
- __epsilon__: float for the epsilon you want to apply
- __query__: Function to apply over private data (see: [Query](../../Private/Query)). This parameter is optional and             the identity function (see: [IdentityFunction](../../Private/Query/#identityfunction-class)) will be used             if it is not provided.

__References__

- [The algorithmic foundations of differential privacy](
   https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/dp_mechanism.py#L141)</span>
### ExponentialMechanism

```python
shfl.differential_privacy.dp_mechanism.ExponentialMechanism(u, r, delta_u, epsilon, size=1)
```


Implements the exponential mechanism differential privacy defined by Dwork in 
"The algorithmic Foundations of Differential Privacy".

__Arguments:__

- __u__: utility function with arguments x and r. It should be vectorized, so that for a         particular database x, it returns as many values as given in r.
- __r__: array for the response space.
- __delta_u__: float for the sensitivity of the utility function.
- __epsilon__: float for the epsilon you want to apply.
- __size__: integer for the number of queries to perform at once. If not given it defaults to one.

__References__

- [The algorithmic foundations of differential privacy](
   https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    