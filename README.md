# Aml_2019 Coursework, Group 9, Part I
Experiments with Gradient Descent
---

In Machine Learning, there is a loss/cost function in each model, which can gives the performances of the model with different values of each parameters. A better performance usually means a smaller loss in the loss function. Gradient descent is the fundamental method which can lead us to approach to the minimum of the loss function and find the coeffients that reduce the loss to the minimum. 

By starting from a randomly point as our initialization, we will find its gradient, which is also known as the rate of change given a unit change in coefficient. Then, we take a step to "downhill"/negative gradient direction and move to the next point on the loss curve. In the process, the step size we take to the next point is also known as learning rate. If the gradient of the standing point is steep, there will be large decrease in the loss function for each step we take. Alternatively, if the gradient is small, then a large step can only cause a small change in the loss. After taking some steps, we find that the loss cannot be improved by a lot anymore, meaning that we reach the convergence or approach to the minimum. In the plain vanilla gradient descent, the step size(eta) is constant and the change in the loss in each step will depend on how steep the gradient of the point is.

## Examples: Three-Hump Camel Function
---
<p align="left">
  <img width="400" height="300" src="https://github.com/luluytzhang/Aml_2019_Group9/blob/master/1.jpg"/400/300>
  <img width="400" height="300" src="https://github.com/luluytzhang/Aml_2019_Group9/blob/master/2.jpg"/400/300>
</p>
Figure: 3D-Plot of Three-Hump Camel Function & Loss Path of Plain Vanilla Gradient Descent

---

| Initial Point |Step Size (eta) | No. of Iterations | Minimum Loss |  Minimum values(x)  |
| --- | ---| ---| --- | --- |
| [1,1]      |eta=0.001       |        3908       |   3.67e-07   |[-0.000260600061, 0.000629308619] |
| [1,1]      |eta=1           |        2          |   30.85876   | [2.5772800000000005, 2.8]        |
| [2,2]      |eta=0.001       |        4546       |  0.29863870  |[1.7475011736524, -0.87325172472] |
| [2,2]      |eta=0.1         |        37         |   1.90e-07   |[-0.000187361385, 0.000452342408] |

Chart: Changing initial guess point and step size to see the differences of results.

---
**Conclusion:** The global minimum of Three-Hump Camel Function is [0,0] with another two local minimum points. We firstly set step size as 0.001, it takes 3908 steps to reach the global minimum point. However, when change the initial point, it stops at the local minimum point. By increasing the step size, it goes back to global minimum point. This is due to the issue of plain vanilla gradient decent. When the step size is too small, the process will stop at saddle point where the gradient is zero but it is not the minimum. However, if we increase the step size to 1, we can see from the result that the process diverges and stops at anypoint where has a zero gradient.

## Momentum Gradient Descent & Nesterov's Accelerated Gradient Descent(NAG)
The challenge of plain vanilla gradient descent is to determine how large the step size we need to take. If the step size is too small, the processing time will be long. On the other hand, if the size is large, it may diverge. Therefore, in this project, we discuss another two types of gradient descent, Momentum and Nesterov's Accelerated Gradient(NAG). 

| Gradient Type | No. of Iterations | Minimum Loss |  Minimum values(x)  |
| ---| ---| --- | --- |
| Plain Vanilla | 3908 | 3.67e-07 | [-0.000260600061, 0.000629308619] |
| Momentum | 100 | 0.000723240 | [-0.02033033485, 0.010489517057]|
| NAG | 288 | 3.62e-07 | [-0.000259188399, 0.0006243440254] |

**Conclusion:** We experiment with another two types of gradient descent, Momentum and Nesterov's Accelerated Gradient (NAG). There is a new item is included in both methods, a hyper prameter(alpha), representing how much we take part previous gradient. In our project, we choose alpha=0.9. Based on plain vanilla, momentum method takes direction of previous gradients in the updating process at each iteration, so the process will reach minimum faster and can avoid stucking at the saddle point. NAG is very close to Memntum. It calculates gradient ahead with interim parameters, and then update the values in the same way of Momentum. We can see that both methods take less steps to reach minimum comparing with plain vanilla and the NAG's result is more accurate and closer to the global minimum point comparing with Momentum.


