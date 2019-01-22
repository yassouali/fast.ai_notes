### How to download data

Kaggle CLI is a good tool to use when we are downloading from Kaggle. After installing Kaggle-CLI, we can download the datasets directly from the terminal.
```
kg download -u <username> -p <password> -c <competition>
```
A quite interesting addon is `CurlWget`, which gives us a curl command we can past to the terminal the use directly.

#### Create Submission file for Kaggle
Let's take for example the dog vs cats kaggle competition, we first run our model on the test set and obtain by default the log probabilities using the fastai library, so we exponentiate them to get back the probabilities:
```python
log_preds,y = learn.TTA(is_test=True)
probs = np.exp(log_preds)
```

And we create a dataframe to store them, and write them to disk using the correct format:
```python
df = pd.DataFrame(probs)
df.columns = data.classes
```
One additionnal column we need to add, that is required by kaggle, is the ID of the images / examples in the test set. So we insert a new column at position zero named ‘id’ and remove first 5 and last 4 letters since we just need ids in this case.

```python
df.insert(0,'id', [o[5:-4] for o in data.test_ds.fnames])
```
And the results are:

|  | id        | cat      | dogs     |
|--|:---------:|:--------:|:--------:|
|0 | /828      | 0.000005 | 0.999994 |
|1 | /10093    | 0.979626 | 0.013680 |
|2 | /2205     | 0.999987 | 0.000010 |
|3 | /11812    | 0.000032 | 0.999559 |
|4 | /4042     | 0.000090 | 0.999901 |

After that we can write the dataframe to disk, to inspect it further and upload it to kaggle, we note that with large files compression is important to speedup work

```python
SUBM = f'{PATH}sub/'
os.makedirs(SUBM, exist_ok=True)
df.to_csv(f'{SUBM}subm.gz', compression='gzip', index=False)
```

If we're working in a distant server, we can use `FileLink` to back a URL that you can use to download onto your computer. For submissions, or file checking etc.

```python
FileLink(f'{SUBM}subm.gz')
```

#### Softmax loss

In a classification task, the goal is to learn a mapping h:X→Y, for either a:

* Binary vs multiclass: In binary classification, |Y|=2 (e.g, a positive category, and a negative category). In Multiclass classifcation, |Y|=k for some k∈N.

* Single-label vs multilabel: This refers to how many possible outcomes are possible for a single example x∈X. This refers to whether the chosen categories are mutually exclusive, or not. For example, if we are trying to predict the color of an object, then we're probably doing single label classification: a red object can not be a black object at the same time. On the other hand, if we're doing object detection in an image, then since one image can contain multiple objects in it, we're doing multi-label classification.

Softmax is used only for multi-class cases, in which the inputs is one of M classes, for multilabel classification, where each output might belong to multiple classes, we can use sigmoid unit in the output, giving us and output of size batch_size x number_classes, just like softmax, the only difference is that time the ouputs are not normlized, but each element is between [0,1], so then use a simple binary cross with the labels. but we also can train a multilabel classifier with tanh & hinge, by just treating the targets to be in {-1,1}. Or even sigmoid & focal loss with {0, 1} with problems with 1:1000 training imbalance, when we want to reduce the effect of the frequent class and increase the loss of the unfrequent ones:

$$F L \left( p _ { t } \right) = - \left( 1 - p _ { t } \right) ^ { \gamma } \log \left( p _ { t } \right)$$

Here we added a new term: (1- pt)^γ, the log loss is calculates as for binary cross entropy, and then the resuls is multiplied by this term, setting γ > 0 reduces the relative loss for well-classified examples (pt > 0.5), putting more focus on hard, misclassified examples. they also add a wight alpha.

```python
def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
   epsilon = 1e-8
   y_pred += epsilon

   cross_entropy = -y_true * np.log(y_pred)
   weight = np.power(1 - y_pred, gamma) * y_true
   flocal_loss = cross_entropy * weight * alpha

   reduce_fl = np.max(flocal_loss, axis=-1)
   return reduce_fl
```

#### Difference in log base for cross entropy calcuation

Log base e and log base 2 are only a constant factor off from each other :

$$\log_{n}X = \frac{\log_{e} X} {\log_{e} n}$$

Therefore using one over the other scales the entropy by a constant factor. When using log base 2, the unit of entropy is bits, where as with natural log, the unit is nats.  One isn't better than the other. It's kind of like the difference between using km/hour and m/s.  It is possible that log base 2 is faster to compute than the logarithm. However, in practice, computing cross-entropy is pretty much never the most costly part of the algorithm, so it's not something to be overly concerned with. In practice the logarithm used is the natural logarithm (base e).

#### Structured and unstructured data

Structured data is data that can be both syntactically and semantically described by a straightforward format description. Data in CSV files, XML files, JSON files, email headers, and to some extent HTML is structured, because once you have the format specifier for a particular file, you can easily identify specific values in the data and what they mean semantically. A whole lot of business-gathered data is in lists, tables, or other structured formats.

Unstructured data is data that is in a more ambiguous format. It may (or may not) be in a well-defined syntactical format, such as a video or audio image, but its semantic contents are not obvious from its format. Natural-language text, audio and video files, etc are much harder to “mine” than structured data, so analyzing them requires a more statistical approach to figuring what the data means semantically.

For structured data, to use it with neural networks, we need excesive data preprocessing, such as transforming all the features represented as strings as one hot vectors, and replace all the nan values with the mean and add a new binary column to specify if the nan values for each given feature, we can also add more features based on the dates and collected from 3rd parties. the data are then normelized and passed through the network.

