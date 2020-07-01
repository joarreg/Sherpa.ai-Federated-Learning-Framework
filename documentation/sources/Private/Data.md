<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/data.py#L33)</span>
### DataAccessDefinition

```python
shfl.private.data.DataAccessDefinition()
```


Interface that must be implemented in order to define how to access the private data.

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/data.py#L51)</span>
### UnprotectedAccess

```python
shfl.private.data.UnprotectedAccess()
```


This class implements access to data without restrictions, plain data will be returned.

----

<span style="float:right;">[[source]](https://github.com/sherpaai/Sherpa.FL/blob/master/shfl/private/data.py#L4)</span>
### LabeledData

```python
shfl.private.data.LabeledData(data, label)
```


Class to represent labeled data

__Arguments:__

- __data__: Features representing a data sample
- __label__: Label for this sample
    