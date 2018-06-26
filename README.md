<br>
<h2 align="center">Monkey Species Classification </h2>
<p align="center"> 
  Project carried out by José Jesús Torronteras Hernández, for University of  Rome  "La Sapienza"
</p>

 
## Table of contents
- [Quick start](#quick-start)
- [Execution](#execution)
- [Model](#Model)
- [Results](#results)


## Quick start

It is necessary to have installed: Python 3.5.2 and dowload [Monkey Species Dataset](https://www.kaggle.com/slothkong/10-monkey-species) in Input Folder

The present code has been developed under python3. The simplest way to run the program is creating a virtual environment, for this it is necessary to have installed [pip](https://pypi.python.org/pypi/pip) and [virtualenv](https://github.com/pypa/virtualenv).

```bash
# We create the environment
$ virtualenv --python python3 monkey-species-classification
# We activate the environment
$ source monkey-species-classification/bin/activate
# Install all the necessary python packages
$ cd monkey-species-classification
$ git clone https://github.com/xexuew/Monkey-Species-Classification.git .
$ pip3 install -r requirements.txt
```

## Execution

I have designed a main program that executes the programs that the user wants.

```python3
$ python3 main.py
```

## Model

![Model](https://github.com/xexuew/Monkey-Species-Classification/blob/master/assets/model_plot.png)


## Results

![results](https://github.com/xexuew/Monkey-Species-Classification/blob/master/assets/VGG_results.png)
![Confussion](https://github.com/xexuew/Monkey-Species-Classification/blob/master/assets/confussion.png)