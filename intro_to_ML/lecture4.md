
# LECTURE 4

#### Hyper parameters of interest in Random Forests

**1- `set_rf_samples`**

<p align="center"> <img src="../figures/bootstrap.png" width="350"> </p>

Determines how many rows are in each tree. So before we start a new tree, we either bootstrap a sample (i.e. sampling with replacement from the whole dataset) or we pull out a subsample of a smaller number of rows and then we build a tree from there.

Step 1 is we have our whole big dataset, we grab a few rows at random from it, and we turn them into a smaller dataset. From that, we build a tree.

<p align="center"> <img src="../figures/sampling.png" width="500"> </p>

Assuming that the tree remains balanced as we grow it, and assuming we are growing it until every leaf is of size one, the tree will be log2(20K) layers deep. Same thing for the number of leaf nodes, we'll have 20K leaf nodes, one for each examples. We have a linear relationship between the number of leaf nodes and the size of the sample. So when you decrease the sample size, there are less final decisions that can be made. Therefore, the tree is going to be less rich in terms of what it can predict because it is making less different individual decisions and it also is making less binary choices to get to those decisions.

Setting RF samples lower is going to mean that you overfit less, but it also means that we are going to have a less accurate individual tree model. We are trying to do two things when we build a model with bagging. One is that each individual tree/estimator is as accurate as possible (so each model is a strong predictive model). But then across the estimators, the correlation between them must be as low as possible so that when we average them out together, we end up with something that generalizes. By decreasing the `set_rf_samples`, we are actually decreasing the power of the estimator and increasing the correlation

**2- `min_samples_leaf`**

Before, we assumed that `min_samples_leaf=1`, if it is set to 2, the new depth of the tree is log2(20K/2) = log2(20K) - 1. Each time we double the `min_samples_leaf` , we are removing one layer from the tree, and halving the number of leaf nodes (i.e. 10k). The result of increasing `min_samples_leaf` is that now each of our leaf nodes contains more than one example, so we are going to get a more stable average that we are calculating in each tree. We have a little less depth (i.e. we have less decisions to make) and we have a smaller number of leaf nodes. So again, we would expect the result of that node would be that each estimator would be less predictive, but the estimators would be also less correlated. So this might help us avoid overfitting.

**3- `max_features `**

At each split, it will randomly sample columns (as opposed to `set_rf_samples` where we pick a subset of rows for each tree). With `max_features=0.5`, at each split, we’d pick a different subset of the features, here we pick half of them. The reason we do that is because we want the trees to be as rich as possible. Particularly, if we were only doing a small number of trees (e.g. 10 trees) and we picked the same column set all the way through the tree, we are not really getting much variety in what kind of things it can find. So this way, at least in theory, seems to be something which is going to give us a better set of trees by picking a different random subset of features at every decision point.

### Model Interpretation

#### One hot encoding

For categorical variables represented as codes (number in the range of the cardinality of the variables / number of categoreis), it might take a number of splits for the tree to get the important category, say the category we're interested in is situated in position 4, and we hae 6 categories, so we'll need to split to set the desired one, and this increases and the computation needed and the complexity of the tree. One solution is to replace the one column of categories with the a number of columns depending on the cardinality of the features, say for 6 categories we'll have 6 one hot encoding column to replace it, this might help the tree, but most importantly, we can use to find the important features that were hidden prior to this.

In code this is done using the function `proc_df`, which automatically replaces all the column with a cardinality less than `max_n_cat` with one hot columns.

```python
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
```

Interestingly, it turns out that before, it said that enclosure was somewhat important. When we do it as one hot encoded, it actually says `Enclosure_EROPS w AC` is the most important thing. So for at least the purpose of interpreting the model, we should always try one hot encoding quite a few of your variables. We can try making that number as high as you can so that it doesn’t take forever to compute and the feature importance doesn’t include really tiny levels that aren’t interesting.

<p align="center"> <img src="../figures/one_hot_encoding.png" width="600"> </p>

#### Removing redundant variables

We’ve already seen how variables which are basically measuring the same thing can confuse our variable importance. They can also make our random forest slightly less good because it requires more computation to do the same thing and there’re more columns to check. So we are going to do some more work to try and remove redundant features using “dendrogram”. And it is kind of hierarchical clustering.

**Side notes:**

- Correlation coefficients are used in statistics to measure how strong a relationship is between two variables. There are several types of correlation coefficient.Population correlation coefficient: $r_ { x y } = \frac { \sigma _ { x y } } { \sigma _ { x } \sigma _ { y } }$
- Rank correlation, is the correlation between the ranks of tha variables, and not their values.
- Spearman’s  correlation  coefficient  is a  statistical  measure  of  the  strength  of  a  monotonic relationship between paired data. In a sample it is denoted by rs and is by  design constrained between -1 and 1. And  its  interpretation  is  similar  to  that  of  Pearsons,  e.g. the closer  is  to  the  stronger  the  monotonic  relationship.  Correlation  is  an  effect  size  and  so  we  can  verbally  describe  the  strength  of  the  correlation  using  the  following  guide  for  the  absolute value of :

After applying the hierarchical clustering to the data frame, and drawing the following dendogram:

<p align="center"> <img src="../figures/dendogram.png" width="600"> </p>

We see that a number of variables are strongly correlated, For example `saleYear` and `saleElapsed` are measuring basically the same thing (at least in terms of rank) which is not surprising because saleElapsed is the number of days since the first day in my dataset so obviously these two are nearly entirely correlated.

One thing we can do to assure that these variables are very similar, is to remove one of them and see if the models performances stayed the same or not.

```python
def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, 
           max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_

for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 
          'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))
```

And we see that the results (saleYear 0.889037446375 saleElapsed 0.886210803445 ...) are very similar to the baseline (0.889).

#### Partial dependence

After ploting the values of `YearMade`, we see that a lot of them were made in year 1000, which means that they were made before a certain year and we didn't track them, so we can just grab things that are more recent, let's say, they were made before 1930, and use `ggplot`, which was originally was an R package (GG stands for the Grammar of Graphics). The grammar of graphics is this very powerful way of thinking about how to produce charts in a very flexible way. We use it to plot a realationship between to features, which will other wise give a very dense scatter plot.

```python
ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')
```

We using only 500 points from the data frame, and ploting `YearMade` against `SalePrice`. aes stands for “aesthetic”, and we add standard error (se=True) to show the confidence interval of this smoother. and the dark line is the resulst of using `loess`, which is a locally weighted regression.

We see a dipe in price, which might be missliding, and we could come to the conclusion that the bulldozers made in the period 1990 - 1997 are not good, and should not be sold, but their might be other factors in play, like a recession. One alternative is to replace the column `YearMade` with a given year, say starting from 1960 up to 2000, and for each time, we predict the prices using our trained Random Forest for each point in our data frame (in this examples 500), so we'll end up with 500 predictions for each Year (the ones in Blue), and by taking their average we can see that indeed the average price goes up when the `YearMade` increases, this is done using `pdp` - partial dependence plot.

```python
def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)

plot_pdp('YearMade')
```
<p align="center"> <img src="../figures/pdp_plot.png" width="700"> </p>

This partial dependence plot is something which is using a random forest to get us a more clear interpretation of what’s going on in our data. The steps were:

* First of all look at the future importance to tell us which things do we think we care about.
* Then to use the partial dependence plot to tell us what’s going on on average.

We can also use `pdp` to find the price based on two variables (say `YearMade` and `saleElapsed`), or we can also cluster the 500 prediction (blue lines) to obtain the most import ones.

<p align="center"> <img src="../figures/pdp_plot_2vars.png" width="700"> </p>
