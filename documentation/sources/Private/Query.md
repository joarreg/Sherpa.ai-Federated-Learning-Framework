<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/query.py#L5)</span>
## Query class

```python
shfl.private.query.Query()
```


This class represents a query over private data. This interface exposes a method receiving
data and must return a result based on this input.


---
## Query methods

### get


```python
get(data)
```



Receives data and apply some function to answer it.

__Arguments:__

- __data__: Data to process

__Returns:__

- __answer__: Result of apply query over data
    
----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/query.py#L24)</span>
## IdentityFunction class

```python
shfl.private.query.IdentityFunction()
```


This function doesn't transform data. The answer is the data.

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/query.py#L32)</span>
## Mean class

```python
shfl.private.query.Mean()
```


Implements mean over data array.
