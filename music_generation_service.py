from predictions import predict
from midi_utilities import convert_midi_to_wav, change_instruments, notes_to_chords
from utilities import get_file_name_for_saving
from glob import glob
import jsonpickle
import config

def run_generation(generator, requested_operations):

    file_name, *_ = get_file_name_for_saving()

    available_operations = {
    }

    # Turn this into its own service file, called music generator or something
    file_name, output_midi_filename, *_ = predict(generator, file_name)
    print(file_name, output_midi_filename)

    current_file_name = output_midi_filename

    for operation_key in available_operations:
        if operation_key in requested_operations:

            operation_value = requested_operations[operation_key]
            operation = available_operations[operation_key]

            current_file_name = operation(current_file_name, file_name, operation_value)
        else:
            print("TEST", operation_key)

    sound_font = ""
    output_file_path = convert_midi_to_wav(current_file_name, file_name, sound_font)
    return output_file_path

def get_wav_by_name(file_name):
    file_path = config.RESULTS_DIR + "/%s*.wav" % file_name
    wav_file = glob(file_path)[0]
    return wav_file