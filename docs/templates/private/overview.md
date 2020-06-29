# Private

This package contains most of the core elements of the framework that are used in almost every code that you will write 
with Sherpa.ai Federated Learning Framework. 

Maybe the most important element in the framework is the [DataNode](../data_node). A DataNode represents a 
device or element containing private data. In real world scenarios this data is typically property of a user or company.
 This data is private and access must be defined to be used. In this framework the definition is done through 
 [DataAccessDefinition](../data/#dataaccessdefinition-class), a function that is applied to data before share private
 information with someone out of the node. There is an special class of access where there is no access restrictions to 
 the private data, [UnprotectedAccess](../data/#unprotectedaccess-class).