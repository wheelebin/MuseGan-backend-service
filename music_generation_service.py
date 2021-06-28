from predictions import GenOne, GenTwo, GenThree, GenFour
from midi_utilities import convert_midi_to_wav, change_instruments, notes_to_chords, set_drums, stretch_midi_to_length, get_midi_by_genre, stretch_midi_to_length
from utilities import get_file_name_for_saving
from glob import glob
import jsonpickle
import config


class MusicGenKing:
    def __init__(self, generator_file_path):
        #self.genOne = GenOne(generator_file_path)
        #self.genTwo = GenTwo()
        self.genThree = GenThree()
        self.genFour = GenFour()

    def run_generation(self, gen_type, requested_operations):

        # Tempo needs to be set pre prediction
        tempo = None
        if "set_bpm" in requested_operations:
            tempo = requested_operations["set_bpm"]
            requested_operations.pop("set_bpm", None)

        genre = None
        if "genre" in requested_operations:
            genre = requested_operations["genre"]
            midi_from_genre = get_midi_by_genre(genre) 
            if midi_from_genre == None:
                return None

        file_name, output_midi_filename = self.predict(gen_type, tempo, genre)

        current_file_name = self.run_midi_mutation_ops(
            file_name, output_midi_filename, requested_operations
        )
        output_file_path = self.convert_midi_to_wav(current_file_name, file_name)

        return file_name, output_file_path

    def predict(self, gen_type, tempo=None, genre=None):

        file_name, *_ = get_file_name_for_saving()

        #if gen_type == "ai1":
        #    file_name, output_midi_filename, *_ = self.genOne.predict(file_name, tempo)
        #elif gen_type == "ai2":
        #    file_name, output_midi_filename, *_ = self.genTwo.predict(file_name, tempo)
        if gen_type == "ai3":
            file_name, output_midi_filename, *_ = self.genThree.predict(file_name, genre)
        elif gen_type == "ai4":
            file_name, output_midi_filename, *_ = self.genFour.predict(file_name, genre)
        else:
            return None

        return (file_name, output_midi_filename)

    def run_midi_mutation_ops(
        self, file_name, output_midi_filename, requested_operations
    ):

        available_operations = {
            "change_instruments": change_instruments,
            "add_drums": set_drums,
            "add_chords": notes_to_chords,
            # set_bpm // This runs pre prediction
            "modify_length": stretch_midi_to_length,
        }

        current_file_name = output_midi_filename

        for operation_key in available_operations:
            if operation_key in requested_operations:

                operation_value = requested_operations[operation_key]
                operation = available_operations[operation_key]

                current_file_name = operation(
                    current_file_name, file_name, operation_value
                )

        return current_file_name

    def convert_midi_to_wav(self, current_file_name, file_name):
        # Set this in req
        # config.SOUNDFONTS_DIR + "/kit3.sf2"
        sound_font = ""
        output_file_path = convert_midi_to_wav(current_file_name, file_name, sound_font)
        return output_file_path


def predict():
    return 1


def run_generation(generator, requested_operations):

    a = MusicGenKing()

    print(requested_operations)

    file_name, output_midi_filename, *_ = a.predict("ai2")

    current_file_name = a.run_midi_mutation_ops(file_name, output_midi_filename)
    output_file_path = a.convert_midi_to_wav(current_file_name, file_name)

    return output_file_path


def get_wav_by_name(file_name):
    file_path = config.RESULTS_DIR + "/%s*.wav" % file_name
    wav_file = glob(file_path)[0]
    return wav_file
