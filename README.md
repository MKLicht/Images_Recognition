# Images Recognition
 Image recognition algorithms for MNIST database and animal face image database.

## Handwritten Digit Image Recognition
### Test
`/MNIST_fully` contains a fully connented network, and `MNIST_class` contains a convolutional network.

If you want to test your own data within the fully connected network pretrained models, place your images within `/MNIST_fully/test_images`.

Then run
```python
cd MNIST_fully
python main.py
```
If you want to test your own handwritten data within the convolutional network, download and extract the pretrained model from [Google drive](https://drive.google.com/file/d/1OQwCyiPbUTKBISxQ6hRafAVJaPt9Hybt/view?usp=sharing),
then place the `mnist_net_1000.pth` into `/MNIST_class/models` directory.

Place your own data within `/MNIST_class/test_images` directory. 

Then run
```python
cd MNIST_class
python main.py
```
### Train
If you want to train the fully connected network, run
```python
cd MNIST_fully
python mnist.py
```
You can check the loss function and accuracy in training process by the graphs in `/MNIST_fully/train_figure` directory after training.

If you want to train the convolutional network, run
```python
cd MNIST_class
python mnist.py
```
You can check the loss function and accuracy in training process by the graphs in `/MNIST_class/train_figure` directory after training.

## Animal classification
### Test
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/17KXKKV9MEIjy44kON67x0GN6LTTFJIAi/view?usp=sharing), then place `aninet_net_1000.pth` whthin the `/Ani_class/models` directory.

Place your own images whthin the `/Ani_class/test_images` directory, run
```python
cd Ani_class
python main.py
```

### Train
Download the dataset from this [link](https://www.kaggle.com/datasets/andrewmvd/animal-faces?resource=download), place the `/afhq` directory within the `/Ani_class` directory. Then run
```python
cd Ani_class
python aninet.py
```
The loss function and accuracy graphs are generated in `/Ani_class/train_figure` after training.
