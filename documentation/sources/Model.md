<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/model/model.py#L4)</span>
## TrainableModel class

```python
shfl.model.model.TrainableModel()
```


Interface of the models that can be trained. If you want to use a model that is not implemented
in the framework you have to implement a class with this interface.


---
## TrainableModel methods

### train


```python
train(data, labels)
```



Method that train the model

__Arguments:__

- __data__: Data to train the model
- __labels__: Label for each train element
    
---
### predict


```python
predict(data)
```



Predict labels for data

__Arguments:__

- __data__: Data for predictions

__Returns:__

- __predictions__: Matrix with predictions for data
    
---
### evaluate


```python
evaluate(data, labels)
```



This method must return the performance of the prediction for those labels

__Arguments:__

- __data__: Data to be evaluated
- __labels__: True values of data
    
---
### get_model_params


```python
get_model_params()
```



Gets the params that define the model

__Returns:__

- __params__: Parameters defining the model
    
---
### set_model_params


```python
set_model_params(params)
```



Update the params that define the model

__Arguments:__

- __params__: Parameters defining the model
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/model/deep_learning_model.py#L6)</span>
## DeepLearningModel class

```python
shfl.model.deep_learning_model.DeepLearningModel(model, batch_size=None, epochs=1, initialized=False)
```


This class offers support for Keras and tensorflow models.

__Arguments:__

- __model__: Compiled model, ready to train
- __batch_size__: batch_size to apply
- __epochs__: Number of epochs
- __initialized__: Indicates whether the model is initialized or not (default False)
    