<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_distribution/data_distribution.py#L10)</span>
## DataDistribution class

```python
shfl.data_distribution.data_distribution.DataDistribution(database)
```


Abstract class for data distribution

__Arguments:__

- __database__: Database to distribute. (see: [Databases](../../Databases))
    

---
## DataDistribution methods

### get_federated_data


```python
get_federated_data(num_nodes, percent=100, weights=None)
```



Method that split the whole data between the established number of nodes.

__Arguments:__

- __num_nodes__: Number of nodes to create
- __percent__: Percent of the data (between 0 and 100) to be distributed (default is 100)
- __weights__: Array of weights for weighted distribution (default is None)

__Returns:__

  * **federated_data, test_data, test_label**
    
---
### make_data_federated


```python
make_data_federated(data, labels, num_nodes, percent, weights)
```



Method that must implement every data distribution extending this class

__Arguments:__

- __data__: Array of data
- __labels__: Labels

num_nodes : Number of nodes

- __percent__: Percent of the data (between 0 and 100) to be distributed (default is 100)
- __weights__: Array of weights for weighted distribution (default is None)

__Returns:__

- __federated_data__: Data for each client
- __federated_label__: Labels for each client
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_distribution/data_distribution_iid.py#L6)</span>
## IidDataDistribution class

```python
shfl.data_distribution.data_distribution.IidDataDistribution(database)
```


Implementation of an independent and identically distributed data distribution

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_distribution/data_distribution_non_iid.py#L7)</span>
## NonIidDataDistribution class

```python
shfl.data_distribution.data_distribution.NonIidDataDistribution(database)
```


Implementation of a non-independent and identically distributed data distribution
