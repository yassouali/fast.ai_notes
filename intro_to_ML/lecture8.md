### Normalize

Normalizing data is subtracting out the mean and dividing by the standard deviation.

Does it matter in Random Forests to normalize? Not really, the key is that when we are deciding where to split, all that matters is the order. Like all that matters is how they are sorted, so if we subtract the mean divide by the standard deviation, they are still sorted in the same order. Random forests only care about the sort order of the independent variables. They don’t care at all about their size. So that’s why they’re wonderfully immune to outliers because they totally ignore the fact that it’s an outlier, they only care about which one is higher than what other thing. So this is an important concept. It doesn’t just appear in random forests. It occurs in some metrics as well. For example, area under the ROC curve completely ignores scale and only cares about sort. Also dendrograms and Spearman’s correlation only cares about order, not about scale. So random forests, one of the many wonderful things about them are that we can completely ignore a lot of these statistical distribution issues. But we can’t for deep learning because deep learning, we are trying to train a parameterized model. So we do need to normalize our data. If we don’t then it’s going to be much harder to create a network that trains effectively.

So we grab the mean and the standard deviation of our training data and subtract out the mean, divide by the standard deviation, and that gives us a mean of zero and standard deviation of one.

```python
mean = x.mean()
std = x.std()

x=(x-mean)/std
```
Now for our validation data, we need to use the standard deviation and mean from the training data. We have to normalize it the same way.

### Look at the data

In any sort of data science work, it's important to look at your data, to make sure you understand the format, how it's stored, what type of values it holds, etc. To make it easier to work with, let's reshape it into 2d images from the flattened 1d format.

Helper methods
```python
def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)

def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')
```

### NN as universal approximators [Link](http://neuralnetworksanddeeplearning.com/chap4.html)

### Softmax and log softmax

There are a number of advantages of using log softmax over softmax including practical reasons like improved numerical performance and gradient optimization. These advantages can be extremely important for implementation especially when training a model can be computationally challenging and expensive. At the heart of using log-softmax over softmax is the use of log probabilities over probabilities, which has nice information theoretic interpretations.

When used for classifiers the log-softmax has the effect of heavily penalizing the model when it fails to predict a correct class. Whether or not that penalization works well for solving your problem is open to your testing, so both log-softmax and softmax are worth using.

Gradient methods generally work better optimizing log p(x) (-log p goes to inf if the p goes to zero, and one if p=1 and it is the correct class) than p(x) because the gradient of log p(x) is generally more well-scaled. That is, it has a size that consistently and helpfully reflects the objective function's geometry, making it easier to select an appropriate step size and get to the optimum in fewer steps.

Compare the gradient optimization process for p(x)=exp(−x^2) and f(x) = log p(x) = −x^2. At any point x, the gradient of f(x) is f′(x)=−2x. If we multiply that by 1/2, we get the exact step size needed to get to the global optimum at the origin, no matter what x is. This means that we don't have to work too hard to get a good step size (or "learning rate"). No matter where our initial point is, we just set our step to half the gradient and we'll be at the origin in one step. And if we don't know the exact factor that is needed, we can just pick a step size around 1, do a bit of line search, and we'll find a great step size very quickly, one that works well no matter where x is. This property is robust to translation and scaling of f(x). While scaling f(x) will cause the optimal step scaling to differ from 1/2, at least the step scaling will be the same no matter what x is, so we only have to find one parameter to get an efficient gradient-based optimization scheme.

In contrast, the gradient of p(x) has very poor global properties for optimization. We have p′(x)=f′(x)p(x)=−2x exp(−x2). This multiplies the perfectly nice, well-behaved gradient −2x with a factor exp(−x2) which decays (faster than) exponentially as x increases. At x=5, we already have exp(−x2)=1.4 10^−11, so a step along the gradient vector is about 10^−11 times too small. To get a reasonable step size toward the optimum, we'd have to scale the gradient by the reciprocal of that, an enormous constant ∼10^11. Such a badly-scaled gradient is worse than useless for optimization purposes - we'd be better off just attempting a unit step in the uphill direction than setting our step by scaling against p′(x)! (In many variables p′(x) becomes a bit more useful since we at least get directional information from the gradient, but the scaling issue remains.)

In general there is no guarantee that log p(x) will have such great gradient scaling properties as this toy example, especially when we have more than one variable. However, for pretty much any nontrivial problem, log p(x) is going to be way, way better than p(x). This is because the likelihood is a big product with a bunch of terms, and the log turns that product into a sum. Provided the terms in the likelihood are well-behaved from an optimization standpoint, their log is generally well-behaved, and the sum of well-behaved functions is well-behaved. By well-behaved we mean f′′(x) doesn't change too much or too rapidly, leading to a nearly quadratic function that is easy to optimize by gradient methods. The sum of a derivative is the derivative of the sum, no matter what the derivative's order, which helps to ensure that that big pile of sum terms has a very reasonable second derivative!