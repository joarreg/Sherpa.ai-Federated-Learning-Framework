<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/node.py#L6)</span>
### DataNode

```python
shfl.private.node.DataNode()
```


This class represents an independent data node.

A DataNode has its own private data and provides methods
to initialize this data and access to it. The access to private data needs to be configured with an access policy
before query it or an exception will be raised. A method to transform private data is also provided. This is
a mechanism that allows data preprocessing or related task over data.

A model (see: [Model](../../Model)) can be deployed in the DataNode and use private data
in order to learn. It is assumed that a model is represented by its parameters and the access to these parameters
must be also configured before queries.

----

### set_private_data


```python
set_private_data(name, data)
```



Creates copy of data in private memory using name as key. If there is a previous value with this key the
data will be overridden.

__Arguments:__

- __name__: String with the key identifier for the data
- __data__: Data to be stored in the private memory of the DataNode
    
----

### configure_data_access


```python
configure_data_access(name, data_access_definition)
```



Adds a DataAccessDefinition for some concrete private data.

__Arguments:__

- __name__: String with the key identifier for the data
- __data_access_definition__: Policy to access data (see: [DataAccessDefinition](../DataAccessDefinition))
    
----

### configure_model_params_access


```python
configure_model_params_access(data_access_definition)
```



Adds a DataAccessDefinition for model parameters.

__Arguments:__

- __data_access_definition__: Policy to access parameters (see: [DataAccessDefinition](../DataAccessDefinition))
    
----

### apply_data_transformation


```python
apply_data_transformation(private_property, federated_transformation)
```



Executes FederatedTransformation (see: [Federated Operation](../Federated Operation)) over private date.

__Arguments:__

- __private_property__: String with the key identifier for the data
- __federated_transformation__: Operation to execute (see: [Federated Operation](../Federated Operation))
    
----

### query


```python
query(private_property)
```



Queries private data previously configured. If the access didn't configured this method will raise exception

__Arguments:__

- __private_property__: String with the key identifier for the data
    
----

### query_model_params


```python
query_model_params()
```



Queries model parameters. By default the parameters access is unprotected but access definition can be changed

----

### set_model_params


```python
set_model_params(model_params)
```



Sets the model to use in the node

__Arguments:__

- __model_params__: Parameters to set in the model
    
----

### train_model


```python
train_model(training_data_key)
```



Train the model that has been previously set in the data node

__Arguments:__

- __training_data_key__: String identifying the private data to use for this model. This key must contain             LabeledData (see: [Data](../../Data))
    
----

### predict


```python
predict(data)
```



Uses the model to predict new data

__Arguments:__

- __data__: Data to predict
    
----

### evaluate


```python
evaluate(data, labels)
```



Evaluates the performance of the model

__Arguments:__

- __data__: Data to predict
- __labels__: True values of data
    