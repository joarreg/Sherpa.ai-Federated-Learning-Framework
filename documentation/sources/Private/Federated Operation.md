<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/federated_operation.py#L78)</span>
## FederatedData class

```python
shfl.private.federated_operation.FederatedData()
```


Class representing data across different data nodes.

This object is iterable over different data nodes.


---
## FederatedData methods

### add_data_node


```python
add_data_node(data)
```



This method adds a new node containing data to the federated data

__Arguments:__

- __data__: Data to add to this node
    
---
### num_nodes


```python
num_nodes()
```



__Returns:__

- __num_nodes__: The number of nodes in this federated data.
    
---
### configure_data_access


```python
configure_data_access(data_access_definition)
```



Creates the same policy to access data over all the data nodes

__Arguments:__

- __data_access_definition__: (see: [DataAccessDefinition](../Data/#dataaccessdefinition))
    
---
### query


```python
query()
```



Queries over every node and returns the answer of every node in a list

__Returns__

- __answer__: List containing responses for every node
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/federated_operation.py#L6)</span>
## FederatedDataNode class

```python
shfl.private.federated_operation.FederatedDataNode(federated_data_identifier)
```


This class represents a [DataNode](../DataNode) in a FederatedData. Extends DataNode allowing
calls to methods without explicit private data identifier, assuming access to the federated data.

__Arguments:__

- __federated_data_identifier__: identifier to use in private data

When you iterate over [FederatedData](./#federateddata-class) the kind of DataNode that you obtain is a     FederatedDataNode.

__Example:__


```python
# Definition of federated data from dataset
database = shfl.data_base.Emnist()
iid_distribution = shfl.data_distribution.IidDataDistribution(database)
federated_data, test_data, test_labels = iid_distribution.get_federated_data(num_nodes=20, percent=10)

# Data access definition and query node 0
federated_data.configure_data_access(UnprotectedAccess())
federated_data[0].query()
```
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/federated_operation.py#L136)</span>
## FederatedTransformation class

```python
shfl.private.federated_operation.FederatedTransformation()
```


Interface defining the method for applying an operation over [FederatedData](./#federateddata-class)


---
## FederatedTransformation methods

### apply


```python
apply(data)
```



This method receives data to be modified and performs the required modifications over it.

__Arguments:__

- __data__: The object that has to be modified
    
----

## federate_array


```python
shfl.private.federated_operation.federate_array(array, num_data_nodes)
```



Creates [FederatedData](./#federateddata-class) from an indexable array.

The array will be divided using the first dimension.

__Arguments:__

- __array__: Indexable array with any number of dimensions
- __num_data_nodes__: Number of nodes to use

__Returns__

- __federated_array__: [FederatedData](./#federateddata-class) with an array of size len(array)/num_data_nodes         in every node
    
----

## apply_federated_transformation


```python
shfl.private.federated_operation.apply_federated_transformation(federated_data, federated_transformation)
```



Applies the federated transformation over this federated data.

Original federated data will be modified.

__Arguments:__

- __federated_data__: [FederatedData](./#federateddata-class) to use in the transformation
- __federated_transformation__: [FederatedTransformation](./#federatedtransformation-class) that will be applied         over this data
    