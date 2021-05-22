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