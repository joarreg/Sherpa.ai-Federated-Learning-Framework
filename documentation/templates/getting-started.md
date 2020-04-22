# Getting started with Sherpa.FL

Sherpa.FL is a python framework that provides an environment to develop research in the fields of private and 
distributed machine learning. The framework is designed with the goal of provide a set of tools allowing users to 
create and evaluate different aspects of this kind of algorithms with minimum code effort.

The main big topics covered at the moment in the framework are federated learning and differential privacy. This 
techniques can be used together in order to increase the privacy of a federated learning algorithm. 

Even if are mainly interested in differential privacy a good point to start with Sherpa.FL is to follow the following 
[notebook](https://github.com/sherpaai/Sherpa.FL/blob/master/notebooks/basic_concepts.ipynb) where are explained the 
main concepts that are used all the time in the tutorials and in the documentation.

The notebooks assume familiarity with python and some of the most popular libraries like numpy or keras/tensorflow. The 
documentation is divided following the different packages in the framework. In every section there is a brief 
introduction of the module with the purpose and some illustrative examples. In many cases the documentation links with 
a notebook illustrating the use of the different modules and classes.

* Package [private](../private/overview) contains most of the core elements of the framework that are used in almost every 
code that you will write with Sherpa.FL.
* Package [data_base](../databases) introduces some datasets to work with.
* Package [data_distribution](../data_distribution) provides some modules to distribute data among nodes.
* Package [federated_aggregator](../federated_aggregator) has different algorithms to aggregate models.
* Package [learning_approach](../learning_approach) defines the communication and the kind of relationships between 
nodes.
* Package [model](../model) provides a set of common models that you might want to use.
* Package [differential_privacy](../differential_privacy/overview) introduces different differential privacy algorithms to 
protect data privacy when this must be shared.


