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
(1.7.0+cu101 - https://download.pytorch.org/whl/torch_stable.html)

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
- Make sure all outputed files are uniquely named
- Make sure these functions work
    - Gluing and trimming midi for a certain predetermined time in seconds, implemented the ability to use templates in the file name, so only generated midis can be glued in one of 4 ways
    
    - Save/download end-points for both .mid and .wav 

    - ai3 & ai4 are not ai but they are "reverse playback" & "tonal inversion" (Don't know which order).
        - Message from Upwork "This is not ai in fact, just reverse playback and tonal inversion in the MuseGAN_torch_main file these functions"
        - But if this is only reverse playback or tonal inversion than the bellow sentance in the tech docs he sent me don't make sense
            - "The genre field allows you to select the genre of generated music, (this option is only available when working with ai3 and ai4"

    - Maybe the service which is responisble for generating a file name should be some service, which takes care of loading, saving, naming and etc safeley and can be used [DONE]
    - Service for orchestrating the neccesary actions[DONE]
    - Replacing notes with chords, you can set in which track or tracks to change and select a major or minor triad [DONE]
    - Changing instruments by tracks [DONE]
    - Convert midi to wav using various sound fonts in sf2 format. [DONE]


# Notes
- Each generation request is unique and delivered songs can't be modified but they can be downloaded again.
- The genre portion of the bellow generation proccess is removed until I figure it out but here are soem notes
    * Genre will most likley dictate which pre-trained model (Different trained models for different genre's) is used so it would need to one of the first operations
- We will mainly focus on returning the WAV but if the user wants to save the midi as well they can request it with the file name
- Generation proccess
    * path, filename = getFilename()
    * output_npz_file = predict(generator, filename)
    * current_file = output_npx_file

    * requested_operations = {
        change_instruments: {
            1: 0,
            2: 0,
            3: 27
        },
        add_drums: True,
        add_chords: True,
        set_bpm: 100,
        modify_length: 260,
        tone_invert: True,
        invert_midi: True,
    }

    * available_operations = {
        change_instruments: change_instruments,
        add_drums: add_drums,
        add_chords: add_chords,
        set_bpm: set_bpm,
        modify_length: modify_length,
        tone_invert: tone_invert,
        invert_midi: invert_midi,
        }

    * for operation in available_operations:
        if operation in requested_operations:
            operation_value = requested_operations[operation]
            current_file = operation(operation_value)

    * output_wav_file = convert_midi2_wav(current_file)
    * return output_wav_file


- WRITE OPERATIONS TO HAPPEN WHEN THEY NEED TO IN A LOOP SO YOU ONLY NEED TO LOOP ONCE AND SAVE ONCE
- (Do this in your own project)


# Firebase
- How to auth (https://firebase.google.com/docs/auth)
    - This is more of an overview of how to implement firebase auth into your client (ios, android, web (front-end) and etc...)
    - Quotes from above
        - To sign a user into your app, you first get authentication credentials from the user.
            - These credentials can be the user's email address and password, or an OAuth token from a federated identity provider.

        - Then, you pass these credentials to the Firebase Authentication SDK.
            - Our backend services will then verify those credentials and return a response to the client.

        - After a successful sign in, you can access the user's basic profile information

        - You can also use the provided authentication token to verify the identity of users in your own backend services.

- Verify id tokens (https://firebase.google.com/docs/auth/admin/verify-id-tokens)
    - This is what I'll be doing in the back-end to create and manage users and etc
    
    - Take the uid and try to get the user, if it dosen't exist create it.
    - Than do whatever you need to do with the uid / user.