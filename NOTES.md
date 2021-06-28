Music Generators

    * AI1 (PyTorch) & AI2 (Tensorflow) are really neurons
    * The folder with genres is only for AI3 (Inversion) & AI4 (Rover)

    * Inversion can ve done in defferent combinations, which gives us unlimited possibilities of combininations
    
    * When the rover is generated, make it an inversion as well
        - I think this means that rover should always be ran through an inversion before converted to a wav

        - The are some peculiarities from which part of the midi is not converted to a rover,
          and some are not converted to tonal inversion.
          In this case you can take another file from the database and convert it.

          - If part of midi is not converted to inversion or rover, delete it, take new midi from and try that one (!)

          - How will I figure out if parts of the midi is not converted to rover or tonal inversion (?)

    * Handling midi's

        - Drop folders if they have same name, keeping the one with the most midi files

        - Dump all midi's into one folder
            - Taking into account the genre, use genre as the root folder (rock, pop)

        - Change name of each midi to random guid for example 97r6c0b87

        - If midi is less than 10kb than delete it from folder

        - If midi is less than 3 minutes than double it

        - Do we have a sorted version of the genre midi files?

    * AI1. PyTorch
    * AI2. Tensorflow
    * AI3. Inversion
    * AI4. Rover (E.g music before (C)re, (D)mi, (E), the rover is me, re, befpre EDC )

    

TODO: 

* Write file to go through all genres using glob
    files = glob.glob('./jazz/**/*.mid', recursive=True)
* check if match against above midi rules, throw away if not
