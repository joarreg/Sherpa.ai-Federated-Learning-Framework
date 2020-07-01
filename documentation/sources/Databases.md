<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_base/data_base.py#L31)</span>
## DataBase class

```python
shfl.data_base.data_base.DataBase()
```


Abstract class for data base.

Load method must be implemented in order to create a database able to     interact with the system, in concrete with data distribution methods     (see: [Data Distribution](../../Data Distribution)).

Load method should save data in the protected Attributes:

__Attributes:__

* **_train_data, _train_labels, _validation_data, _validation_labels, _test_data, _test_labels**

__Properties:__

- __train__: Returns train data and labels
- __validation__: Returns validation data and labels
- __test__: Returns test data and labels
- __data__: Returns train data, train labels, validation data, validation labels, test data and test labels
    

---
## DataBase methods

### shuffle


```python
shuffle()
```



Shuffles all data

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_base/emnist.py#L7)</span>
## Emnist class

```python
shfl.data_base.emnist.Emnist()
```


Implementation for load EMNIST data

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_base/fashion_mnist.py#L7)</span>
## FashionMnist class

```python
shfl.data_base.fashion_mnist.FashionMnist()
```


Implementation for load FASHION-EMNIST data

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/data_base/california_housing.py#L5)</span>
## CaliforniaHousing class

```python
shfl.data_base.data_base.CaliforniaHousing()
```


This database loads the     [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)
from sklearn, mainly for regression tasks.

----

## extract_validation_samples


```python
shfl.data_base.data_base.extract_validation_samples(data, labels, dim)
```



Method that randomly choose the validation data from data and labels.

__Arguments:__

- __data__: Numpy matrix with data for extract the validation data
- __labels__: Numpy array with labels
- __dim__: Size for validation data

__Returns:__

- __new_data__: Data, labels, validation data and validation labels
    