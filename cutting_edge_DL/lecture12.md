## Generative Adversarial Networks (GANs)

In this lecture we are going to focus on generative adversarial networks also known as GANs, and more Wasserstein GAN paper which was heavily influenced by the deep convolutional generative adversarial network (DCGaN).

The most important part of the paper is the algorithm:

<p align="center"> <img src="../figures/wgan_algo.png" width="500"> </p>

The basic idea of GaNs, is an having two models, a generative model that generates fake samples, that need to be quite similar to the real samples of the dataset, and a critic, which given a sample, need to differentiate between fake and real samples, we first began by training the generative model for a few epochs, calculating the loss and updating its parameters, and then alternating between training the critic and the generative models.

So the main task is to create the models, which are going to be simple CNN networks, a classification network for the critic and a CNN with an encoding the decoding blocks for the generator, and the losses for both networks, training GaNs is quite difficult, and the real addition WGaN paper brings, is the new loss fuction and how to control the gradients to ensure training convergence.

### WGaN
We are going to the LSUN dataset classification dataset and use the bedroom category of images to create real images ones, so we are going to download the dataset, unzip it, and convert it to jpg files (using the `lsun-data.py` script):

```python
curl 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag=latest&category=bedroom&set=train' -o bedroom.zip
unzip bedroom.zip
pip install lmdb
python lsun-data.py {PATH}/bedroom_train_lmdb --out_dir {PATH}/bedroom
```
and here is the content of the `lsun-data.py` script:

```python
def export_images(db_path, out_dir, flat=False):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in tqdm(cursor):
            key = key.decode()
            if not flat: image_out_dir = join(out_dir, '/'.join(key[:3]))
            else: image_out_dir = out_dir
            if not exists(image_out_dir): os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.jpg')
        with open(image_out_path, 'wb') as fp: fp.write(val)
```

And then we create the paths to be used for load the stored images, and store then file in the form of CSV, 

```python
PATH = Path('data/lsun/')
IMG_PATH = PATH/'bedroom'
CSV_PATH = PATH/'files.csv'
TMP_PATH = PATH/'tmp'
TMP_PATH.mkdir(exist_ok=True)
```

As always, it is much easier to go the use CSV files when it comes to handling the data, so we generate a CSV with the list of files that we want, and a fake label “0” because we don’t really have labels for these at all. One CSV file contains everything in that bedroom dataset, and another one contains random 10%. It is nice to do that because then we can most of the time use the sample when we are experimenting because there is well over a million files even just reading in the list takes a while.

```python
files = PATH.glob('bedroom/**/*.jpg')

with CSV_PATH.open('w') as fo:
    for f in files: fo.write(f'{f.relative_to(IMG_PATH)},0\n')

# Optional - sampling a subset of files
CSV_PATH = PATH/'files_sample.csv'

files = PATH.glob('bedroom/**/*.jpg')

with CSV_PATH.open('w') as fo:
    for f in files:
        if random.random()<0.1: 
            fo.write(f'{f.relative_to(IMG_PATH)},0\n')
```
>>>>>>> refs/remotes/origin/master
