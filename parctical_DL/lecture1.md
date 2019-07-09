
### Jupyter Notebook shortcuts

Here is a list of shortcuts to use on a daily basis:

**general shortcuts**

Note: The Jupyter help menu shows all the shorcuts as a capital letter, but we should be using the lowercase letters:

| Command         | Result                                        |
|-----------------|:---------------------------------------------:|
| h               | Show help menu                                |
| a               | Insert cell above                             |
| b               | Insert cell below                             |
| m               | Change cell to markdown                       |
| y               | Change cell to coden                          |

**Code shortcuts**

When we are editing the Python code, there are a number of shortcuts to help to speed up the typing:

| Command         | Result                                        |
|-----------------|:---------------------------------------------:|
| Tab             | Code completion                               |
| Shift+Tab       | Shows method signature and docstring          |
| Shift+Tab (x2)  | Opens documentation pop-up                    |
| Shift+Tab (x3)  | Opens permanent window with the documentation |

**Notebook configuration**

* Reload extensions: `%reload_ext autoreload` Whenever a module we are using gets updated, the Jupyter IPython’s environment will be able to reload that for us, depending on how we configure it.

* Autoreload configuration: `%autoreload 2`, In combination with the previously mentioned reload extensions, this setting will indicate how we want to reload modules. We can configure this in three ways:

   * 0 - Disable automatic reloading;
   * 1 - Reload all modules imported with %import every time before executing hte python code typed;
   * 2 - Reload all modules (except those excluded by %import) every time before executing the Python code typed.

* Matplotlib inline: `%matplotlib inline`, This option will output plotting commands inline within the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

**Secure access via port forwarding**

When running the Jupyter Notebooks on our own server, we need to find a way to securely access it.

```
# ssh -L local_port:remote_host:remote_port remote_machine_ip
$ ssh -L 8888:localhost:8888 my_vm_ip
```

This command will map our local 8888 port to localhost:8888 on the remote machine, allowing us to securely connect to the in using directly the localhost adress in the browser.

To maintain an open connection / usage of our terminal without a closing of the connection, we can use Tmux / Screen / Boyubu.

### Looking at the data

As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
1. A few correct labels at random
2. A few incorrect labels at random
3. The most correct labels of each class (i.e. those with highest probability that are correct)
4. The most incorrect labels of each class (i.e. those with highest probability that are incorrect)
5. The most uncertain labels (i.e. those with probability closest to 0.5).

```python
def rand_by_mask(mask):
   return np.random.choice(np.where(mask)[0], 4, replace=False)

def rand_by_correct(is_correct):
   return rand_by_mask((preds == data.val_y)==is_correct)

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

-> plot_val_with_title(rand_by_correct(True), "Correctly classified")
-> plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
```


### Data transformations

These transformations can vary for each architecture but usually entail one or more of these:
* resizing: each images gets resized to the input the network expects;
* normalizing: data values are rescaled to values between 0 and 1; $y = \frac { x - x _ { \min } } { x _ { \max } - x _ { \min } }$
* standardizing: data values are rescaled to a standard distribution with a mean of 0 and a standard deviation of 1; where μ is the mean and σ is the standard deviation. $y = \frac { x - \mu } { \sigma }$. where μ is the mean and σ is the standard deviation.


### References:
* [FastAi lecture 1 notes](https://www.zerotosingularity.com/blog/fast-ai-part-1-course-1-annotated-notes/)