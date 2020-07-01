<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/federated_aggregator/federated_aggregator.py#L4)</span>
## FederatedAggregator class

```python
shfl.federated_aggregator.federated_aggregator.FederatedAggregator(percentage=None)
```


Interface for Federated Aggregator

__Arguments:__

- __percentage__: Percentage of total data in each client
    

---
## FederatedAggregator methods

### aggregate_weights


```python
aggregate_weights(clients_params)
```



Abstract method that aggregates the weights of the client models. 

__Returns:__

- __aggregated_weights__: Aggregated weights
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/federated_aggregator/avgfed_aggregator.py#L6)</span>
## AvgFedAggregator class

```python
shfl.federated_aggregator.federated_aggregator.AvgFedAggregator(percentage=None)
```


Implementation of Average Federated Aggregator. It only uses a simple average of the parameters of all the models

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/federated_aggregator/weighted_avgfed_aggregator.py#L6)</span>
## WeightedAvgFedAggregator class

```python
shfl.federated_aggregator.federated_aggregator.WeightedAvgFedAggregator(percentage=None)
```


Implementation of Average Federated Aggregator. The aggregation of the parameters is based in the number of data     in every node.
