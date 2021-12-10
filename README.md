# SeqTO_SynRM
Implementation of Sequence based Topology Optimization method (SeqTO-v1). 

Based on the following paper:

[https://ieeexplore.ieee.org/abstract/document/8960651](https://ieeexplore.ieee.org/abstract/document/8960651)

A. Khan, C. Midha and D. A. Lowther, "**Sequence-Based Environment for Topology Optimization**," in IEEE Transactions on Magnetics, vol. 56, no. 3, pp. 1-4, March 2020, Art no. 7510904, doi: 10.1109/TMAG.2019.2960731.

---
A new Topology Optimization methodology is presented for the application of electromagnetic designs, which is termed as SeqTO (Sequence Based Topology Optimization). Topology Optimization is an important engineering tool for obtaining novel designs and is successfully applied in engineering domains of structural, mechanical and aerospace.

The proposed TO methodology transforms the material distribution problem into a movement sequence search problem for a controller moving in the design space. This method enforces connectivity between the cells in the discretized design domain that contain the same material. This removes the need of any filtering or smoothing of the optimal result to obtain a manufacturable design. 

![Fig 1](https://user-images.githubusercontent.com/25873155/145520435-897fcd98-d4e5-40c8-a06c-5a19716eed8a.png)


This method leverages a sequence-based environment to impose connectivity on cells containing the same material. The new TO method proposed in this paper neither requires any filtering or smoothing technique nor any modification to the optimization objective function for obtaining manufacturable optimal solutions.

Furthermore, such a method facilitates the application of Reinforcement Learning algorithms. A test problem of rotor design optimization of a Synchronous Reluctance Machines (SynRM) is presented with the code.

Design Domain: 

<img src="https://user-images.githubusercontent.com/25873155/145520904-9d5bf37d-667c-44e8-a601-e8d959902d28.png" width="250" height="250">

SynRM with optimized rotor:

<img src="https://user-images.githubusercontent.com/25873155/145520842-ce61307d-4cc7-4293-991e-858369dc4ad3.png" width="250" height="250">

---

**Dependencies**
1. Siemens MAGNET v2020.2
2. Python 3
3. Tensorflow 2
