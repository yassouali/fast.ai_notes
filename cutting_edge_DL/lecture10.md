<!-- vscode-markdown-toc -->
* 1. [IMDb dataset](#IMDbdataset)
* 2. [Data preprocessing](#Datapreprocessing)
* 3. [Pre-training](#Pre-training)
	* 3.1. [wikitext103 conversion](#wikitext103conversion)
* 4. [Language model](#Languagemodel)
	* 4.1. [A deeper look](#Adeeperlook)
	* 4.2. [The language model](#Thelanguagemodel)
	* 4.3. [Measuring accuracy](#Measuringaccuracy)
* 5. [Text Classification](#TextClassification)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Lecture 10: Natural Language Processing

In this lecture, which is a continuation of lesson 4, we'll try to use a similar approach we used in computer vision, which consisted of taking pre-trained models in image classification, and replace the head to adapat the output to our application, we'll do a similar thing in NLP, we'll first train our model in language model setting on WikiTex dataset, fine tune it as a language model on a target dataset, in our case, we'll use text classification in IMDB, and then add a custom head for text classification, to train the custom head and then fine-tune the whole model with different learning rates to avoid catastrophic forgetting.

##  1. <a name='IMDbdataset'></a>IMDb dataset

This is a dataset of movie reviews, containing positive and negative (75 000 reviews for training and 25 000 for classification, 50 000 of the training set are unlabeled) and some unlabeled examples, in lesson 4 we used torchtext, but this library is quite slow partly because it’s not doing parallel processing and partly it’s because it doesn’t remember what we did last time and it does it all over again from scratch, so in this lecture we'll a new library called `fastai.text`. `fastai.text` is a replacement for the combination of torchtext and `fastai.nlp`.

First we'll go through all the examples in three classes (that are in folders of the same names) and add each one example and its label to the training and validation sets.

```python
CLASSES = ['neg', 'pos', 'unsup']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')

len(trn_texts),len(val_texts)
# (75000, 25000)
```
And we end up with there are 75,000 in train, 25,000 in test. 50,000 in the train set are unsupervised, and we won’t actually be able to use them when we get to the classification, the next step is to shuffle them with a random permutation (of the index only)

```python
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
```

Now we have our texts and labels sorted, we can create a dataframe from them [24:07]. Why are we doing this? The reason is because there is a somewhat standard approach starting to appear for text classification datasets which is to have the training set as a CSV file with the labels first, and the text of the NLP documents second. So it basically looks like this:

As a last step, to avoid doing the same thing every time, we'll create two directories, one for the unsupervised examples (to be used only for training the language model) and one for the classification, and store in them a pandas data frames of these examples, and also a file called `classes.txt` which just lists the classes, this way of saving these files are somehow the standard, and was used in a recent paper by Yann Lecun.

First we'll save the examples to be used in the classification task, i.e. the labeled examples.

```python
PATH=Path('data/aclImdb/')
CLAS_PATH=Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

col_names = ['labels','text']
df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)
# Saving the classification file
df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)
df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)
# Saving the classes
(CLAS_PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)
```

For the our language model, we'll start by training our model for predicting the next word, so as a language model, and we can use all the text, from both classification and unsupervised partitions to fine tune our model already trained on wikitext, for this we need to split our dataset into training set and validation set using scikit learn's `model_selection`, in which we'll used 10% of the whole dataset as our validation set.

```python
trn_texts,val_texts = sklearn.model_selection.train_test_split( np.concatenate([trn_texts,val_texts]), test_size=0.1)

len(trn_texts), len(val_texts)
# (90000, 10000)
```

And now we can save the training and validation examples to be used in training the language model:

```python
LM_PATH=Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)

df_trn = pd.DataFrame({'text':trn_texts, 'labels': [0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels': [0]*len(val_texts)}, columns=col_names)

df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)
```

##  2. <a name='Datapreprocessing'></a>Data preprocessing

In data preprocessing, we need to start by tokenizing our corpus; **Tokenization** means at this stage, for a document (i.e. a movie review), we have set of words that we'd like to turn it into a list of tokens to give a standardized set. For example, `don’t` we want it to be `do` and `n’t`, we probably want full stop to be a token, and so forth. Tokenization is something that we passed off to a specific library such as NLTK or `spaCy`, we'll use spacy because it is faster and easier to work with.

First, we clean our corpus using regex to remove some unnecessary character in the dataset, so we call the fixup function which replaces some weird characters that are quite common in some NLP datasets that need to be replaced

```python
re1 = re.compile(r'  +')

df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)

chunksize=24000
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

def fixup(x):
   x = x.replace('#39;', "'").replace('amp;', '&')
        .replace('#146;', "'").replace('nbsp;', ' ')
        .replace('#36;', '$').replace('\\n', "\n")
        .replace('quot;', "'").replace('<br />', "\n")
        .replace('\\"', '"').replace('<unk>','u_n')
        .replace(' @.@ ','.').replace(' @-@ ','-')
        .replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns))
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_
        labels += labels_
    return tok, labels
```

First, we load the dataframes we saved, but this time we use a new parameter `chunksize`, this parameter is used in case we have big dataframe and it used to specify the number of rows per chunk to avoid loading them to the memory at once, and in this case, pandas will return an iterator over the data instead of the datafram data directly, giving us a chunk of the data at each iteration, and for this we use a function in which we iterate over the dataframe in the function `get_all` and call the function `get_texts` that will :

<p align="center"> <img src="../figures/df_imdb.png" width="500"> </p>

* First we load the labels and convert them into integers, which is the first column of our dataframe,
* We add a beginning of stream (BOS) token (any particular strings as long as they don't appear in the corpus) to define the beginning of the text, so that after concatenating all the text together, the model will still be able to detect the beginning of a new article. And given that in some documents we have multiple field, say an introduction, and abstract, so to differentiate between then we add a field token, and so first we add the first field (the second column after the label) with a field of 1, and then we add the rest of the field in the document, each one with a given number (2, 3, ...), in our case, we only have movie reviews, so we only have one field, and the loop will pas skipped, and then we apply the fixup function to all the elements.
* And then we tokenize our text, since spacy does not provide a parallel/multicore version of the tokenizer, the fastai library adds this functionality. This parallel version (a wrapper over the spacy tokenizer) uses all the cores of the CPUs and runs much faster than the serial version of the spacy tokenizer, so we call `proc_all_mp` (process all multi processing, and before that we divide all the text into the number of the threads available in our CPU using `partition_by_cores`) of the fastAi implementation.

One additional trick, is the usage of a token specific to all caps word, so instead of having two different representation of the two words, one regular and an other which is all caps, we can simple lathe case all the words, and for the all caps words we can add a token `t_up`, so now we'll learn that each word that comes with `t_up` before is begin shouted (some other tricks are in the fastai implementation of Tokenizer).

And then we save the results:

```python
np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)

tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')
```

**Numericalizing  the tokens** the last thing we need to do is *numericalize* the token, this is done as follows:

* We make a list of all the words that appear in corpus,
* We keep only the frequent word (say the words that appear at least twice, given that it can be only a spelling mistake, and the bigger the vocabulary the more compitation / time we need), a vocabulary of size 60 000 is quite good,
* We replace every word with its index into that list.

This can be done with Counter class in python; giving us a list of all the unique item in a list with their counts, we create a counter by passing all the words in all the example in the training set (using to for loops), and then given a vocabulary size, we only keep 60 000 most common words in the vocabulary with freaq > 2, and we add the padding and unknown token to replace the removed / infrequent tokens, and we endup with a list items to tokens to *numericalize* the corpus.

```python
freq = Counter(p for o in tok_trn for p in o)
max_vocab = 60000
min_freq = 2

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')
```

We can now go and use this list to create a mapping dict from tokens to numerical values (string to integer). Given that our vocabulary contains only 6000 tokens, and doesn't covert all the them, we need to assign to the words / token we don't find in the list the token `_unk_`, for this we can use a default dict with a lambda function returning 0 if the token is not found in the list, and use the dict to create our train and val sets.

```python
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
```

And like all the previous step, we can go ahead and save the numpy arrays and the vocabulary:

```python
np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))

trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))
```

Now the vocab size is 60,002 and our training language model has 90,000 documents in it.

##  3. <a name='Pre-training'></a>Pre-training

Just like in computer vision, we first start by pre-training the model to do image classification and then fine-tune our model in down stream tasks, in NLP, we first start by pre-training on `wikitext103` to do language modeling. So just like ImageNet allowed us to train things that recognize stuff that kind of looks like pictures, and we could use it on stuff that was nothing to do with ImageNet like satellite images. Why don’t we train a language model that’s good at English and then fine-tune it to be good at movie reviews (image classification).

To pre-train our language model, we first start by using the wikitext dataset, Stephen Merity has already processed Wikipedia, found a subset of nearly the most of it, but throwing away the stupid little articles leaving bigger articles. He calls that wikitext103. Here are the pretrained language model pretrained by the fastai team: [link](http://files.fast.ai/models/wt103/), and now we can create the LSTM model, and load the pretrained weights on fine tune the model on IMBD.

###  3.1. <a name='wikitext103conversion'></a>wikitext103 conversion

So first we grab the wikitext models by the `wget -r` command, it will recursively grab the whole directory which has a few things in it, and create the variables holding the pre-trained language model path.

```python
! wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/

PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'
```

We need to make sure that our language model has exactly the same embedding size, number of hidden, and number of layers as wikitext language model, otherwise we can’t load the weights in.

```python
em_sz, nh, nl = 400, 1150, 3
```

And now we can go ahead the load the wights using `torch.load`, and it return a dictionary containing the name of the layer and a tensor/array of those weights.

The `map_location` argument gives us the possibility to remap the Tensor location at load time. For example this will forcefully remap everything onto CPU: `torch.load('my_file.pt', map_location=lambda storage, location: 'cpu')`, that is not being used in our case.

```python
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
```

Now the problem is that wikitext language model was built with a certain vocabulary which was not the same as ours. The IMDB vocabulary is not the same as wikitext103 model’s vocab. So we need to map one to the other. That’s simple with the dictionary `itos` for the wikitext vocab, mapping the indices to the words.

So first we can use `defaultdict` to do a reverse mapping, with a value -1 in when we access the dictionary, the key (or the word) we're looking for in the wikitext vocab is not there.

```python
itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
```

The objective is to create an embedding matrix of size (vocab size x emd size) for the IMDB model, containing zeros, and replace the rows with the weights of the wikitext model if it is present in the wikitext vocab, so first we start by retrieving the embedding matrix of the wikitext model (`0.encoder.weight`), we then go through every one of the words in our IMDb vocabulary, and we are going to look it up in `stoi2` (string-to-integer for the wikitext103 vocabulary) and see the word is also a part of the wikitext vocab, if it isn't not we'll get -1, and if it is we'll get an index `r` >= 0, and we will just set that row of the embedding matrix to the weight which was stored inside the named element `0.encoder.weight`, and for the words not in wikitext vocab, we set their embedding rows to the mean of all the rows.

```python
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)

new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m
```

We will then replace the encoder weights with new_w turn into a tensor. In the AWD LSTM the decoder (the final layer that turns the prediction back into a vector of probabilities the size of the vocab) uses exactly the same weights (weight tying), so we copy them to the decoder as well. And given that we apply dropout to the embedding matrix, when saving the model, for some reason we end up with a whole new copy, so we do the same thing.

```python
wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))
# T() == torch.from_numpy()
```

##  4. <a name='Languagemodel'></a>Language model

Let’s create our language model. Basic approach we are going to use is we are going to concatenate all of the documents together into a single list of tokens of length 24,998,320. That is going to be what we pass in as a training set.

And like we've seen in the lecture 4, after concatenating the whole documents together, we divide them into 64 (batch size) sections, and each step in the training process we take a number of words from each section, the number of words we take equal to the size of BPTT (back propagation through time), and it varies form an iteration to another to add some randomness, and for each training step, the inpus are matrix of size 64 x BPTT, and the target and of the same size but with a one word offset to the left.

For training, as per usual, after creating the model, setting the dataloaders, loading the weights, creatin the optimizer and setting the dropout rates (for different layers in the model), we call `learner.fit`. We do a single epoch on the last layer which is the embedding layer, given that a lot of words were not found in the wikitext vocabulary, so we need to learn them, then we’ll start doing a few epochs of the full model.

```python
wd=1e-7
bptt=70
bs=52

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl)

learner= md.get_model(opt_fn, em_sz, nh, nl, dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]

learner.freeze_to(-1)
learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)
learner.unfreeze()
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
```

In lesson 4, we had the loss of 4.23 after 14 epochs. In this case, we have 4.12 loss after 1 epoch. So by pre-training on wikitext103, we have a better loss after 1 epoch than the best loss we got for the language model otherwise.

###  4.1. <a name='Adeeperlook'></a>A deeper look

Let's take a deeper look into the language model and dataloaders we're using, for tha dataloader:

```python
class LanguageModelLoader():
    """Returns tuples of mini-batches."""
    def __init__(self, nums, bs, bptt, backwards=False):
        # initializing the values
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        # like we've said ealier, we divide the data over a number of section
        # the number of section equal the batch size
        self.data = self.batchify(nums)
        self.i, self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        """ Iterator implementation"""
        self.i, self.iter = 0,0
        while self.i < self.n-1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                # we randomize the length of BPTT
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        # this refers to the number of iteration
        # each iteration we get #bptt x #bs of data
        return self.n // self.bptt - 1

    def batchify(self, data):
        """splits the data into batch_size counts of sets"""
        # here we take the integer division, so for say 65 000 words, and
        # a batch of 64, we'l have nb = 10 000, and we'll ignoer the last 1000 words
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        # transpose, to have #nb x #bs
        data = data.reshape(self.bs, -1).T
        # when training the model in backwards, we flip the whole 
        # documents
        if self.backwards: data=data[::-1]
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        # in case we're in the end of the data, so take the rest of the words
        seq_len = min(seq_len, len(source) - 1 - i)
        # the targets are offset by one words and are flattened
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)
```

One cool trick in the dataloader above, in the randomization of the BPTT; which is taken from the AWD LSTM. Instead of always grabing 70 words for each epoch, and to add some form of data augmentation (like for batch shuffling in computer vision, which we cannot do in NLP given that the order of the words is important) we can randomly change the sequence length. So 95% of the time, we will use bptt (i.e. 70) but 5% of the time, we’ll use half that. Then the sequence length will be a normally distributed random number with that average and a standard deviation of 5, So the sequence length is seventy-ish and that means every time we go through, we are getting slightly different batches as a little bit of extra randomness.

###  4.2. <a name='Thelanguagemodel'></a>The language model

We are going to create a custom learner, a custom model data class, and a custom model class. 

For the model data class, it doesn’t inherit from anything, we just need pass the training set (a data loader), the validation set, and optionally a test set, plus some other parameters we might need like the bptt, the number of tokens (i.e. the vocab size), the padding index, and the path to where to save the files and models.

Then all of the work happens inside `get_model`, get_model calls `get_language_model` that we will look at later, which just grabs a normal PyTorch nn.Module architecture, and chucks it on GPU. We wrapp the model in a LanguageModel and the LanguageModel is a subclass of BasicModel which almost does nothing except it defines layer groups so we can apply discriminative learning rates where different layers have different learning rates instean of providing learing rates for every layer because there can be a thousand layers. Then finally turn that into a learner by passing the model and it turns it into a learner. In this case, we have overridden learner to use the cross entropy as the default loss funciton.

```python
class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]

class LanguageModelData():
    def __init__(self, path, pad_idx, n_tok, trn_dl, val_dl, test_dl=None, **kwargs):
        self.path,self.pad_idx,self.n_tok = path,pad_idx,n_tok
        self.trn_dl,self.val_dl,self.test_dl = trn_dl,val_dl,test_dl

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = get_language_model(self.n_tok, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
    def _get_crit(self, data):
        return F.cross_entropy
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs, seq_first=True)
    def save_encoder(self, name):
        save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name):
        load_model(self.model[0], self.get_model_path(name))
```

Now let's take look into `get_language_model`, the function that return the AWD LSTM model, we first create a RNN encoder with the parameter we already choose, ie., the size of the embedding matrix and the size of the hidden layers and their number, and the dropout rates, and then we create a decoder wich is a linear decoder (with the weights tied).

```python
def get_language_model(arch:Callable, vocab_sz:int, config:dict=None, drop_mult:float=1.):
    rnn_enc = RNN_Encoder(n_tok, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                 dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(n_tok, emb_sz, dropout, tie_encoder=enc, bias=bias))

class LinearDecoder(nn.Module):
    initrange=0.1
    def __init__(self, n_out, nhid, dropout, tie_encoder=None):
        super().__init__()
        self.decoder = nn.Linear(nhid, n_out, bias=False)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs
```

And here is the implementation of the RNN encoder, or the AWD LSTM, we first create an embedding matrix, and then we call `EmbeddingDropout` to apply the input dropout to the weights of the embedding matrix, and then we create a number of lstm layers, the first one of size emb sz -> hidden sz, and middle layers of size hidden sz -> hidden sz and the last one hidden sz -> emb sz, with dropout in between the layers, and we also apply a weight droptout (drop connect) in which we delete some connection in between the rnn layers, we initialize the embedding weights and also create an dropout layers, that we also apply during traing and not just the init process, and then we can these layers in the forward function.

```python
def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).
    """
    return x.new(*sz).bernoulli_(1-dropout)/(1-dropout)

class EmbeddingDropout(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            size = (self.embed.weight.size(0),1)
            mask = Variable(dropout_mask(self.embed.weight.data, size, dropout))
            masked_embed_weight = mask * self.embed.weight
        else: masked_embed_weight = self.embed.weight

        if scale: masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None: padding_idx = -1

        X = self.embed._backend.Embedding.apply(words,
             masked_embed_weight, padding_idx, self.embed.max_norm,
             self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)
        return X

class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or not self.p: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x

class RNN_Encoder(nn.Module):
    initrange=0.1
    def __init__(self, ntoken, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5):
        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else nhid, (nhid if l != nlayers - 1 else emb_sz)//self.ndir, 1, bidirectional=bidir, dropout=dropouth) for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz,self.nhid,self.nlayers,self.dropoute = emb_sz,nhid,nlayers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])

    def forward(self, input):
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()

        emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1: raw_output = drop(raw_output)
            outputs.append(raw_output)

        self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz)//self.ndir
        return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]
```

###  4.3. <a name='Measuringaccuracy'></a>Measuring accuracy

Normally for language models, we look at at the perplexity which is just 2^(cross entropy). There is a lot of problems with comparing things based on cross entropy loss. and we can also look and the accuracy of the language model predictions.

After training the language model on IMDB dataset, we can go ahead and save it to use it for classification, but no need to save the decoder, which only matters for the langauge model part, and we only need to use the RNN encoder and use its hidden states for text classification.

##  5. <a name='TextClassification'></a>Text Classification

Now we'll use files we already saved for classification, the two dataloader we saved earlier with the classes text file, and also resue the same tokens and vocabulary we've used for the language model, and we also load the document (after the tokenization) and their labels (positive of negative).

```python
# Tokens
tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')
# Vocabulary
itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
# Examples
trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')
# Labels
trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))
```

We create the same parameters for the model (output of size 3, positive, negative and neutral reviews).

```python
bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48

min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1
```

And we create the data loader, the basic idea here is that for the classifier, at each step, we'd like to look at one document and predict if this document positive or negative?, but if we want to pass at each iteration, say 48 documents / reviews and classifiy them all, they need to be of the same size, we can simply randomly shuffle the documnet, and at each time step take a number of documents (= batch size) and padd all them ot get the same number of words, but if they are wildly different lengths, then we're going to be wasting a lot of computation times. If there is one thing that’s 2,000 words long and everything else is 50 words long, that means we end up with 2000 wide tensor, one solution, which is used in torchtext, is to first sort all the documents, and then at each step take the documents of similar lengths, this way we're only going to add the limited amout of padding.

```python
trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)

class TextDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x,self.y,self.backwards,self.sos,self.eos = x,y,backwards,sos,eos

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x),self.y[idx]

    def __len__(self): return len(self.x)
```

And after creating the dataset, we need to turn it into a datalaoder, to turn it into a DataLoader, we simply pass the Dataset to the DataLoader constructor, and at each call we'll get a batch of data, we are going to pass an additinnal parameter, which is a sampler parameter and sampler is a class we are going to define that tells the data loader how to shuffle:

* For validation set, we are going to define something that actually just sorts. It just deterministically sorts it so that all the shortest documents will be at the start, all the longest documents will be at the end, and that’s going to minimize the amount of padding.
* For training sampler, we are going to create a sort-ish sampler which also sorts with a bit of randomness.

SortSampler is class which has a length which is the length of the data source and has an iterator which is simply an iterator which goes through the data source sorted by length (which is passed in as key). For the SortishSampler, it basically does the same thing with a little bit of randomness, first we do a random permutation, divide the indices into sections of 50 * batchsize, and then we sort them based on their length (so the randomness comes from the fact we have 50 section, each one is sorted seperately instead of sorting the whole dataset), and then we place the largest sequence at the beginning so that the dataloader detect the amount of padding to add the rest the sequences of the batch (which are shuffled a second time).

```python
class SortSampler(Sampler):
    def __init__(self, data_source, key):
        self.data_source,self.key = data_source,key
    def __len__(self):
        return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))

class SortishSampler(Sampler):
    """
    Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size. The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing. Without that max key returned first multiple buffers may be allocated when the first created isn't large enough to hold the next in the sequence.
    """
    def __init__(self, data_source, key, bs):
        self.data_source, self.key, self.bs = data_source, key, bs

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([ck[0] for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]),  bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))

trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)
```

The last step is to create the model, we create a encoder of the same type we used ealier (AWD LSTM) and for the decoder, we're doing to take the last hidden activation of the LSTM layer and pass them to two linear layer (first of output of sie 50 and the lasy of 3 which are the number of prediction we want to make).

There is some additional tricks used here for classification:

* Concat pooling: for the decoder we use here (in `PoolingLinearClassifier`)  the mean pool and max pool of the three hidden activations of the last layers with the activation of the top layer and then concatenate them and pass them to the linear layer.

* BPT3C: The main difference between `RNN_Encoder` used in language modeling and `MultiBatchRNN` is that the normal RNN encoder for the language model, we could just do bptt chunk at a time, but for the classifier, we need to do the whole document. We need to do the whole movie review before we decide if it’s positive or negative. And the whole movie review can easily be 2,000 words long and we can’t fit 2.000 words worth of gradients in my GPU memory for every single one of the weights. So the idea to go through the whole sequence length one batch of bptt at a time and call super().forward (ie the RNN_Encoder) and grab its outputs and its activation and store them, and for the next call of the RNN encoder, the model is initialized with the final state of the previous batch. 

```python
class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

class MultiBatchRNN(RNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        raw_outputs, outputs = [],[]
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

def get_rnn_classifer(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir, dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops))

m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh,
                      n_layers=nl, pad_token=1,
                      layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
                      dropouti=dps[0], wdrop=dps[1],
                      dropoute=dps[2], dropouth=dps[3])

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
```

And after creating our model, as per usual we are going to use discriminative learning rates for different layers, fine tune our model, and then unfreeze the whole model and training for some additional time, and we end up with state of art resutls in IMDB dataset, and the main advantage is that we can use only a small portion of the dataset to get competitive resutls.

