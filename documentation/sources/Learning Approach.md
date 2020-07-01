<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/learning_approach/learning_approach.py#L4)</span>
## LearningApproach class

```python
shfl.learning_approach.learning_approach.LearningApproach(model_builder, federated_data, aggregator)
```


Abstract class Class used to represent a Learning Approach.

__Arguments:__

- __model_builder__: Function that return a trainable model (see: [Model](../../Model))
- __federated_data__: Federated data to use. (see: [Private](../../Private/Federated Operation))
- __aggregator__: Federated aggregator function (see: [Federated Aggregator](../../Federated Aggregator))
    

---
## LearningApproach methods

### train_all_clients


```python
train_all_clients()
```



Initialize the models of each client and train them

---
### aggregate_weights


```python
aggregate_weights()
```



Calculate aggregated weights and update clients and server models

---
### run_rounds


```python
run_rounds(n, test_data, test_label)
```



Run one more round beggining in the actual state

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/learning_approach/federated_government.py#L4)</span>
## FederatedGovernment class

```python
shfl.learning_approach.learning_approach.FederatedGovernment(model_builder, federated_data, aggregator)
```


Class used to represent Federated Government.
