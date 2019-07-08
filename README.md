# Aml_2019 Coursework, Group 9, Part I
Experiments with Gradient Descent
---

In Machine Learning, there is a loss/cost function in each model, which can gives the performances of the model with different values of each parameters. Gradient decent is the fundamental method which can lead us to approach to the minimum of the loss function and find the coeffients that reduce the loss to the minimum. 

By starting from a randomly point as our initialization, we will find its gradient at this point, which is also known as the rate of change given a unit change in coefficient. The "Gradient" is the steepest direction to reach minimum. Then, we need to take a step to "downhill"/negative gradient direction and move to the next point on the loss curve. We repeat this process until the point slowely decent to the minimum. In the process, the step size we take to the next point is also known as learning rate. If the gradient of the standing point is steep, there will be large decrease in the loss function for each step we take. Alternatively, if the gradient is small, then a large step can only lead to a small change in the loss.
