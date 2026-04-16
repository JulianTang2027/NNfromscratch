# NNfromscratch

This is a manual implementation of a Neural Network + Backpropagation, similar to the Pytorch library. The backpropagation is implemented as a topological sort + each 'node' contains a reference to its parents and a local _backward function. Weights are then accumulated via chain rule and a reverse order of the topological sort.
