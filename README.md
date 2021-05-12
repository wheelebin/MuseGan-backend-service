# Measuring Statistics about Information Sources in MIDI Files
https://nbviewer.jupyter.org/github/craffel/midi-ground-truth/blob/master/Statistics.ipynb

# Measuring the Reliability of MIDI-Derived Annotations
https://nbviewer.jupyter.org/github/craffel/midi-ground-truth/blob/master/Reliability%20Experiments.ipynb

# Optimizing DTW-Based Audio-to-MIDI Alignment and Matching
https://nbviewer.jupyter.org/github/craffel/alignment-search/blob/master/overview.ipynb

# Lakh MIDI Dataset Tutorial
https://nbviewer.jupyter.org/github/craffel/midi-dataset/blob/master/Tutorial.ipynb


# Misc
http://millionsongdataset.com/
https://docs.google.com/document/d/1mZDY0XUZGtemI6zgCSJzQ8MzL1MqPX8WNTkuoNxBd5g/edit
https://towardsdatascience.com/bachgan-using-gans-to-generate-original-baroque-music-10c521d39e52

https://www.intel.com/content/www/us/en/artificial-intelligence/posts/deep-learning-training-and-inference.html

https://console.firebase.google.com/u/1/project/music-generator-9578a/overview

https://www.anaconda.com/

# Repo's
https://github.com/salu133445/musegan

https://salu133445.github.io/pypianoroll
https://salu133445.github.io/muspy

https://github.com/salu133445/ismir2019tutorial

# Libs
https://docs.aiohttp.org/en/stable/index.html
https://fastapi.tiangolo.com/

SCREEN CHEAT SHEET - https://kapeli.com/cheat_sheets/screen.docset/Contents/Resources/Documents/index


# REST API
https://github.com/pytorch/tutorials/blob/master/intermediate_source/flask_rest_api_tutorial.py
https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
https://libiseller.ai/deploying-pytorch-to-production


# Notes
- Don't use pipenv (https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)
    - Here are some alternatives
        * https://virtualenvwrapper.readthedocs.io/en/latest/
        * https://github.com/jazzband/pip-tools
        * https://github.com/python-poetry/poetry


# Other intresting links
Fake anime using DCGAN - https://www.youtube.com/watch?v=cqXKTC4IP10
Building our first simple GAN - https://www.youtube.com/watch?v=OljTVUVzPpM
Sentdex - https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ/videos

https://github.com/onnx/tutorials

https://discuss.pytorch.org/t/runtimeerror-cudnn-error-cudnn-status-not-initialized/115286/2

https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-train-pytorch.md

# This version of torch
(1.7.0+cu101)

## Different states of this application
This project is used to train our model, run predictions on our model and handle restful API requests (responding with predictions).
I need to find a way to easily:
- Train
- Predict without API
- Start web service with access to predictions



## Prerequisites
* Make sure that you have python installed
* Make sure that you have the packages pip, virtualenv installed
* Install Fluidsynth
* Data
    * Download LPD dataset if creating new tensor dataset
    * Download .pt if loading existing tensor dataset for training model 
    * Download checkpoints loading exisiting model
    * Download synth files for midi2audo (Optional, this uses Fluidsynth)


## Installation Steps
1. Clone this repo & cd into it
2. Create a virtualenv & activate it
3. Install pip-tools
4. Run pip-sync to install packages
    * This torch version is installing with cuda, if you don't have a GPU which is supported by cuda than install the CPU version of torch.
5. Run python main.py

# TODO
- Make sure all outputed files are uniquely named´
- Make sure these functions work
    - Gluing and trimming midi for a certain predetermined time in seconds, implemented the ability to use templates in the file name, so only generated midis can be glued in one of 4 ways
    - Replacing notes with chords, you can set in which track or tracks to change and select a major or minor triad
    - Changing instruments by tracks

    - Convert midi to wav using various sound fonts in sf2 format. [DONE]