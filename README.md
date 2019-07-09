# Aml_2019 Coursework, Group 9, Part I
Experiments with Gradient Descent
---

In Machine Learning, there is a loss/cost function in each model, which can gives the performances of the model with different values of each parameters. A better performance usually means a smaller loss in the loss function. Gradient descent is the fundamental method which can lead us to approach to the minimum of the loss function and find the coeffients that reduce the loss to the minimum. 

By starting from a randomly point as our initialization, we will find its gradient, which is also known as the rate of change given a unit change in coefficient. Then, we take a step to "downhill"/negative gradient direction and move to the next point on the loss curve. In the process, the step size we take to the next point is also known as learning rate. If the gradient of the standing point is steep, there will be large decrease in the loss function for each step we take. Alternatively, if the gradient is small, then a large step can only cause a small change in the loss. After taking some steps, we find that the loss cannot be improved by a lot anymore, meaning that we reach the convergence or approach to the minimum. In the plain vanilla gradient descent, the step size(eta) is constant and the change in the loss in each step will depend on how steep the gradient of the point is.

##Examples: Three-Hump Camel Function
---
<p align="left">
  <img width="400" height="300" src="https://github.com/luluytzhang/Aml_2019_Group9/blob/master/1.jpg"/400/300>
  <img width="400" height="300" src="https://github.com/luluytzhang/Aml_2019_Group9/blob/master/2.jpg"/400/300>
</p>
**Figure: Left: the 3D-Plot of Three-Hump Camel Function. Right: The loss path of plain vanilla gradient descent.



The challenge of plain vanilla gradient descent is to determine how large the step size we need to take. If the step size is too small, the processing time will be long. On the other hand, if the size is large, it may diverge. Therefore, in this project, we discuss another two types of gradient descent, Momentum and Nesterov's Accelerated Gradient(NAG). Comparing with plain vanilla, the Momentum takes accounts previous gradients which can accelerate gradient descent. NAG is a further improvement based on Momentum. It performs a lookahead gradient evaluation and then make corrections.


