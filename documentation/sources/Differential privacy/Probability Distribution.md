<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/probability_distribution.py#L5)</span>
## ProbabilityDistribution class

```python
shfl.differential_privacy.probability_distribution.ProbabilityDistribution()
```


Class representing the interface for a probability distribution


---
## ProbabilityDistribution methods

### sample


```python
sample(size)
```



This method must return an array with length "size", sampling the distribution

__Arguments:__

- __size__: Size of the sampling
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/probability_distribution.py#L20)</span>
## NormalDistribution class

```python
shfl.differential_privacy.probability_distribution.NormalDistribution(mean, std)
```


Implements Normal Distribution

__Arguments:__

- __mean__: Mean of the normal distribution.
- __std__: Standard deviation of the normal distribution
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/differential_privacy/probability_distribution.py#L36)</span>
## GaussianMixture class

```python
shfl.differential_privacy.probability_distribution.GaussianMixture(params, weights)
```


Implements the combination of Normal Distributions

__Arguments:__

- __params__: Array of arrays with mean and std for every gaussian distribution.
- __weights__: Array of weights for every distribution with sum 1.

__Example:__


```python
# Parameters for two Gaussian
mu_M = 178
mu_F = 162
sigma_M = 7
sigma_F = 7

# Parameters
norm_params = np.array([[mu_M, sigma_M],
                       [mu_F, sigma_F]])
weights = np.ones(2) / 2.0

# Creating combination of gaussian
distribution = GaussianMixture(norm_params, weights)
```
    