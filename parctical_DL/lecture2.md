### Finding an optimal learning rate

How do we select the “right” learning rate to start training our model? This question is a heavily debated one in deep learning and fast.ai offers a solution based on a paper from Leslie Smith - Cyclical Learning Rates for training Neural Networks.

<p align="center"> <img src="../figures/cyclic_learning_rate.png" width="250"> </p>

The idea of the paper is quite simple:

* start with a small learning rate and calculate the loss;
* gradually start increasing the learning rate and each time, calculate the loss;
* once the loss starts to shoot up again, it is time to stop;
* we select the highest learning rate you can find, where the loss is still crearly improving (steepest decrease of the loss).

<p align="center"> <img src="../figures/lr_finder.png" width="500"> </p>

In the figure above, we see that when we increase the learning rate beyond a certain treshold, the loss goes up, this is when we stop our training and choose the lr that gave us the steepest decrease of the loss.

### learning rate annealing

In training deep networks, it is usually helpful to anneal the learning rate over time. Good intuition to have in mind is that with a high learning rate, the system contains too much kinetic energy and the parameter vector bounces around chaotically, unable to settle down into deeper, but narrower parts of the loss function. Knowing when to decay the learning rate can be tricky: Decay it slowly and we'll be wasting computation bouncing around chaotically with little improvement for a long time. But decay it too aggressively and the system will cool too quickly, unable to reach the best position it can. There are three common types of implementing the learning rate decay:

* **Step decay:** Reduce the learning rate by some factor every few epochs. Typical values might be reducing the learning rate by a half every 5 epochs, or by 0.1 every 20 epochs. These numbers depend heavily on the type of problem and the model. One heuristic you may see in practice is to watch the validation error while training with a fixed learning rate, and reduce the learning rate by a constant (e.g. 0.5) whenever the validation error stops improving.
* **Exponential decay**. has the mathematical form $\alpha = \alpha _ { 0 } e ^ { - k t }$, where α0, k are hyperparameters and t is the iteration number (or the epochs).
* **1/t decay** has the mathematical form $\alpha = \alpha _ { 0 } / ( 1 + k t )$ where a0, k are hyperparameters and t is the iteration number.

<p align="center"> <img src="../figures/lr_annealing.png" width="500"> </p>

In practice, we find that the step decay is slightly preferable because the hyperparameters it involves (the fraction of decay and the step timings in units of epochs) are more interpretable than the hyperparameter k. Lastly, if we can afford the computational budget, err on the side of slower decay and train for a longer time.

### stochastic gradient descent with restarts (SGDR)

With lr annealing, we may find ourselves in a part of the weight space that isn’t very resilient — that is, small changes to the weights may result in big changes to the loss. We want to encourage our model to find parts of the weight space that are both accurate and stable. Therefore, from time to time we increase the learning rate (this is the ‘restarts’ in ‘SGDR’), which will force the model to jump to a different part of the weight space if the current area is “spiky”. Here’s a picture of how that might look if we reset the learning rates 3 times (in this paper they call it a “cyclic LR schedule”):

<p align="center"> <img src="../figures/SGDR.png" width="500"> </p>

From the paper [Snapshot Ensembles](https://arxiv.org/abs/1704.00109)

If we plot the learning rate while using “cyclic LR schedule” we get (the value used in the starting point is the one obtained using learing rate finder:

<p align="center"> <img src="../figures/cyclic_lr.png" width="300"> </p>

The number of epochs between resetting the learning rate is set by cycle length, and the number of times this happens is referred to as the number of cycles. so for epochs=3 and cycle_len=1, we'll do three epochs, one with each cycle.

We can also vary the length of each cycle, where we start with small cycles and the cycle length get multiplied each time.

<p align="center"> <img src="../figures/varying_cyclic_lr.png" width="300"> </p>

In the figure above, we do three cycles, the original length of the cycle is one epoch, and given that we multiply each time the length by two, we'll get 1 + 2 + 4 = 7 epochs in the end.

Intuitively speaking, if the cycle length is too short, it starts going down to find a good spot, then pops out, and goes down trying to find a good spot and pops out, and never actually get to find a good spot. Earlier on, we want it to do that because it is trying to find a spot that is smoother, but later on, we want it to do more exploring. That is why cycle_mult=2 seems to be a good approach.

### Learning rate and fine tuning (differential learning rate annealing)
When using a pretrained model, we can freeze the weights of all the previous layers but the last one, so we only combine the learned features that the model is capable of outputing and use them to classify our inputs.

Now if we want to fine tune the earlier layers to be more specific to our dataset, and only detect the fetures that will actualy help us, we can use diffent learning rates for each layers.

In other words, for cat/dogs dataset, the previous layers have *already* been trained to recognize imagenet photos (whereas our final layers where randomly initialized), so we want to be careful of not destroying the carefully tuned weights that are already there.

Generally speaking, the earlier layers have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at say 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2, which is the same learning rate we got with the learning rate finder `lr=np.array([1e-4,1e-3,1e-2])`.

And we can see that this kind of approach help us find better local minimas with smaller losses.

<p align="center"> <img src="../figures/loss_with_cyclic_lr.png" width="300"> </p>


## How to terain a state of the art classifier:

1. Use data augmentation and pretrained models
1. Use Cyclical Learning Rates to find highest learning rate where loss is still clearly improving
1. Train last layer from precomputed activations for 1-2 epochs
1. Train last layer with data augmentation for 2-3 epochs with stochastic gradient descent with restarts
1. Unfreeze all layers
1. Set earlier layers to 3x-10x lower learning rate than next higher layer
1. Use Cyclical Learning Rates again
1. Train full network with stochastic gradient descent with restarts with varying length until over-fitting


### References:
* [FastAi lecture 2 notes](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-2-eeae2edd2be4)