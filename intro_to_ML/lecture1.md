<!-- vscode-markdown-toc -->
Preliminaries

* 1. [Some notes about Google Colab](#SomenotesaboutGoogleColab)
* 2. [Jupyter tricks](#Jupytertricks)
* 3. [Python tricks](#Pythontricks)

Lesson 1 - Introduction to Random Forests
* 4. [Thecurse of dimensionality?](#Thecurseofdimensionality)
* 5. [No free lunch theorem](#Nofreelunchtheorem)
* 6. [Preprocessing](#Preprocessing)
* 7. [Side note; Skitlearn](#SidenoteSkitlearn)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Preliminaries

##  1. <a name='SomenotesaboutGoogleColab'></a>Some notes about Google Colab

- To install some packages, we can use either `pip` or `apt`:
```python
!pip install -q matplotlib-venn
from matplotlib_venn import venn2
venn2(subsets = (3, 2, 1))
```
```shell
!apt update && apt install -y libfluidsynth1
```
Installing FastAi in Google Colab (example only) [Link](https://gist.githubusercontent.com/gilrosenthal/58e9b4f9d562d000d07d7cf0e5dbd840/raw/343a29f22692011088fe428b0f800c77ccad3951/Fast.ai%2520install%2520script)
For Machine learn course, we need to install the version 0.7 of fastAi, forum response [Link](https://forums.fast.ai/t/importerror-cannot-import-name-as-tensor/25295/3)

To delete some packages, we can do the same thing
```shell
!pip uninstall `fastai`
```

- To add some data to google colab, first we upload it to the Files (we can view the Files angle in the left angle, by clicking a small grey arrow)
After adding the files, we can then import it into the notebook:
````python
# Loadd the Drive helper and mount
from google.colab import drive
# This will prompt for authorization
drive.mount('/content/name_of_the_folder')
# The files will be present in the "/cotent/name_of_the_folder/My Drive"
§ls "/content/name_of_the_folder/ My Drive"
````

* To download the data using Kaggle API, after installing it and exporting the API Keys to the terminal, we download it directly:
````shell
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
kaggle competitions download -c bluebook-for-bulldozers
````


##  2. <a name='Jupytertricks'></a>Jupyter tricks

- In data sience (unlike software engineering), prototyping is very important, and jupyter notebook helps a lot, for example, given a function `display`:

    1. type `display` in a cell and press shift+enter — it will tell we where it came from `<function IPython.core.display.display>`
    2. type `?display` in a cell and press shift+enter — it will show we the documentation
    3. type `??display` in a cell and press shift+enter — it will show we the source code. This is particularly useful for fastai library because most of the functions are easy to read and no longer than 5 lines long.

- `shift + tab` in Jupyter Notebook will bring up the inspection of the parameters of a function, if we hit `shift + tab` twice it'll tell us even more about the function parameters (part of the docs)
- If we put %time, it will tell us how much times it took to execute the line in question.
- If we  run a line of code and it takes quite a long time, we can put %prun in front. `%prun m.fit(x, y)`. This will run a profiler and tells us which lines of code took the most time, and maybe try to change our preprocessing to help speed things up.

##  3. <a name='Pythontricks'></a>Python tricks

1- `getattr` looks inside of an object and finds an attribute with that name, for example to modify the date (datetime format in Pandas), we use it to add more colums in the dataframe depending on the name (Year, Week, Month, Is_quarter_end...), these data attribues are present in field_name.dt (Pandas splits out different methods inside attributes that are specific to what they are. So date time objects will be in `dt`).

2- PEP 8 - Style Guide for Python Code [PEP 8](https://www.python.org/dev/peps/pep-0008/).

3- Reading Data: In case we using large csv files (millions of columns), to be able to load them into the RAM by limiting the amount of space that it takes up when we read in, we create a dictionary for each column name to the data type of that column. 
```python
types = {'id': 'int64',
        'item_nbr': 'int32',
        'store_nbr': 'int8',
        'unit_sales': 'float32',
        'onpromotion': 'object'}

%%time
df_all = pd.read_csv(f'{PATH}train.csv', parse_dates=['date'], dtype=types, infer_datetime_format=True)
```

But generally it is better to use only a subset of the data to explore it, and not loading it all at one go, one possible way to do so is to use the UNIX command `shuf`, we can get a random sample of data at the command prompt and then we can just read that. This also is a good way to find out what data types to use, we read in a random sample and let Pandas figure it out for us. ```shuf -n 5 -o output.csv train.csv```

___

# Lesson 1 - Introduction to Random Forests

Now for the data, we'll use Kaggle data, and there is a trick to download it using cURL from the terminal, by capturing the GET resquest using the inspect element / Network of Firefox browser:

* Press `ctrl + shift + i` to open web developer tool. Go to Network tab, click on Download button, and cancel out of the dialog box. It shows network connections that were initiated. We then right-click on it and select Copy as cURL. Paste the command and add `-o bulldozer.zip` at the end (possibly remove `— — 2.0` in the cURL command), or use the Kaggle API

##  4. <a name='Thecurseofdimensionality'></a>Thecurse of dimensionality?
The more dimension of the data / features we have, it creates a space that is more and more empty, the more dimensions we have, the more all of the points sit on the edge of that space. If we just have a single dimension where things are random, then they are spread out all over. But if it is a square then the probability that they are in the middle means that they cannot be on the edge of either dimension so it is a little less likely that they are not on the edge. Each dimension we add, it becomes multiplicatively less likely that the point is not on the edge of at least one dimension, so in high dimensions, everything sits on the edge. What that means in theory is that the distance between points is much less meaningful. But this turns out not to be the case for number of reasons, like they still do have different distances away from each other. Just because they are on the edge, they still vary on how far away they are from each other and their is some similarities between each point even in these high dimensions.

##  5. <a name='Nofreelunchtheorem'></a>No free lunch theorem
There is a claim that there is no type of model that works well for any kind of dataset. In the mathematical sense, any random dataset by definition is random, so there is not going to be some way of looking at every possible random dataset that is in someway more useful than any other approach. In real world, we look at data which is not random. Mathematically we would say it sits on some lower dimensional manifold. It was created by some kind of causal structure.
 

##  6. <a name='Preprocessing'></a>Preprocessing
Generally, the provided dataset contains a mix of continuous and categorical variables.

* **continuous**: numbers where the meaning is numeric such as price.
* **categorical**: either numbers where the meaning is not continuous like zip code or string such as “large”, “medium”, “small”

One of the most interesting types of features are dates, we can do a lot of feature engineering based on them: It really depends on what we are doing. If we are predicting soda sales in San Fransisco, we would probably want to know if there was a San Francisco Giants ball game that day. In this case, such feature engineering is crucial for getting better performances given that no machine learning algorithm can tell we whether the Giants were playing that day and that it was important.

So before feeding the data to say a random forest, we must first process the dates and also convert all the strings, which are inefficient, and doesn't provide the numeric coding required for a random forest, to a categorical representation for a more rich representation.

For example:
- First, we can change any columns of strings in a panda's dataframe to a column of catagorical values.
```python
for n,c in df.items():
    if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
```
- Pandas then will automatically creates a list of categories (in `df.column.cat.categories`, and their decimal codes in `df.column.cat.codes`)

- An additionnal pre-processing step, is processing missig values (NULL), so for each column, if there is NULL values (`pd.isnull(col).sum` > 0), we create a new column with the same name and null added in the end, and the NULL calues have 1, and the other are zeros (`pd.isnull(col)`), now for the original column, we replace the NULL values by the median of the column `df[name] = col.fillna(col.median())`. This is only done for numerical columns, pandas automaticlly handels categorical data (converted strings in our case), but the NULL in this case equals -1, so we add one to all the columns (`if not is_numerical_dtype(col)`)

**Feather format**: Reading CSV takes about 10 seconds, and processing takes another 10 seconds, so if we do not want to wait again, it is a good idea to save them. Here we will save it in a feather format where we save the preprocessed data directely to disk in exactly the same basic format that it is in RAM. This is by far the fastest way to save something, and also to read it back. Feather format is becoming standard in not only Pandas but in Java, Apache Spark, etc.

##  7. <a name='SidenoteSkitlearn'></a>Side note; Skitlearn

Everything in scikit-learn has the same form and takes the same steps, to be used for random forest (either a regressor for predicting continuous variables `RandomForestRegressor `, or a classifier for predicting categorifcal variables `RandomForestClassifier `)

```python
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
```

And steps are: 
1. Create an instance of an object for the machine learning model, 
2. Call `fit` by passing in the independent variables, giving the columns / features, and the variable to predict, and finnally computes some scores to evaluate the performances of the model.
