import os
from glob import glob
from mido import MidiFile, MidiTrack, tick2second, MetaMessage
from pypianoroll import Multitrack, outputs, track
import music21 as m21
import random
from midi2audio import FluidSynth
from utilities import get_file_name_for_saving


def cut_midi(input_file, output_file, start_time, end_time):
    """
    Cuts midi input_file from start_time to end_time and saves cutted part into
    output_file.
    """
    input_midi = MidiFile(input_file)
    output_midi = MidiFile()
    tempo = 600000
    if input_midi.type == 2:
        print("Can't cut the file")

    # Copying the time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    for original_track in input_midi.tracks:
        for msg in original_track:
            if msg.type == "set_tempo":
                print(msg.tempo)
                tempo = msg.tempo
                break

    for original_track in input_midi.tracks:
        new_track = MidiTrack()
        total_time = 0
        for msg in original_track:
            if msg.type in ["note_on", "note_off"]:
                total_time += tick2second(msg.time, input_midi.ticks_per_beat, tempo)
                if total_time < start_time or total_time > end_time:
                    continue
            new_track.append(msg)
        output_midi.tracks.append(new_track)

    output_midi.save(output_file)


def join_midi(input_files, output_file):
    """
    Takes list of input_files and joins it into one output_file.
    input_files - list of paths to files
    output_file - path to the output file
    """
    output_midi = MidiFile()
    tempo = 600000
    new_tracks = []
    for idx, input_file in enumerate(input_files):
        input_midi = MidiFile(input_file)
        if input_midi.type == 2:
            continue
        # Copying the time metrics between both files
        if idx == 0:
            output_midi.ticks_per_beat = input_midi.ticks_per_beat
            for original_track in input_midi.tracks:
                for msg in original_track:
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                        break

        for track_n, original_track in enumerate(input_midi.tracks):
            if idx == 0:
                new_tracks.append(MidiTrack())
            for msg in original_track:
                new_tracks[track_n].append(msg)
    for track in new_tracks:
        output_midi.tracks.append(track)

    output_midi.save(output_file)


def join_midi_to_length(input_folder, output_file, length):
    """
    Takes midi files from input folder and joins these to length seconds.
    You can use a wildcard in input folder, for example:
    "/content/1*.mid" for files in content folder which names start
    from 1 and end with .mid
    """
    output_midi = MidiFile()
    tempo = 600000
    new_tracks = []
    tracks_length = []
    for idx, file in enumerate(glob(input_folder)):
        input_midi = MidiFile(file)
        if input_midi.type == 2:
            continue
        # Copying the time metrics between both files
        if idx == 0:
            output_midi.ticks_per_beat = input_midi.ticks_per_beat
            for original_track in input_midi.tracks:
                for msg in original_track:
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                        break
        for track_n, original_track in enumerate(input_midi.tracks):
            if idx == 0:
                new_tracks.append(MidiTrack())
                tracks_length.append(0)
            for msg in original_track:
                new_tracks[track_n].append(msg)
                if msg.type in [
                    "note_on",
                    "note_off",
                ]:
                    tracks_length[track_n] += tick2second(
                        msg.time, input_midi.ticks_per_beat, tempo
                    )
                    if tracks_length[track_n] > length:
                        break
    for track in new_tracks:
        output_midi.tracks.append(track)

    output_midi.save(output_file)


def choose_tracks(input_file, output_file, tracklist):
    """
    Example: choose_tracks("/content/2UNLIMITED - Are You Ready For This.mid", "cutted.mid", [0, 1, 2, 6])
    Takes input_file and saves to output_file only tracks with indexes
    from tracklist.
    """
    input_midi = MidiFile(input_file)
    output_midi = MidiFile()

    # Copying the time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    for idx, original_track in enumerate(input_midi.tracks):
        if idx in tracklist:
            new_track = MidiTrack()
            for msg in original_track:
                new_track.append(msg)
            output_midi.tracks.append(new_track)

    output_midi.save(output_file)


def check_tones(input_file):
    """
    Exmaple: min_note, max_note = check_tones("/content/Ievan_Polkka.mid")
    Returns tuple with maximum and minumum note number
    """
    try:
        mid = MidiFile(input_file)
    except:
        return None
    min_note = 128
    max_note = 0
    for track in mid.tracks:
        for message in track:
            if "note" in dir(message):
                if message.note < min_note:
                    min_note = message.note
                if message.note > max_note:
                    max_note = message.note
    return (min_note, max_note)


def open_midi(midi_path, remove_drums=False):
    """
    Opens midi file from midi_path as a music21.Stream object.
    """
    mf = m21.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return m21.midi.translate.midiFileToStream(mf)


def shuffle_midi(in_file, out_file):
    """
    Shuffles measures in a midi file
    """
    s_in = open_midi(in_file)
    try:
        midi_measures = s_in.measures(1, 9999999)
    except:
        print("Can't convert this!")
        return 0
    s_out = m21.stream.Score()

    for part in midi_measures:
        partnew = m21.stream.Part()
        sample_part = random.sample(list(part), len(part))
        for el in sample_part:
            partnew.append(el)
        s_out.append(partnew)

    s_out.write("midi", out_file)


def save_midi_from_npz(npz_filename, midi_filename, n_bins=4):
    """
    Converts npz file into midi. Also cuts every n_bins + 1 measure of midi.
    Every 5th bin is silence in MuseGAN.
    """
    midi_npz = Multitrack(npz_filename)
    midi_npz.write(midi_filename)
    midifile = open_midi(midi_filename)
    try:
        midi_measures = midifile.measures(1, 9999999)
    except:
        print("Can't convert this!")
        os.remove(midi_filename)
        return 0
    midi_cut = m21.stream.Score()

    for part in midi_measures:
        partnew = m21.stream.Part()
        for i, el in enumerate(part):
            if (i + 1) % (n_bins + 1) != 0:
                partnew.append(el)
        midi_cut.append(partnew)
    midi_cut.write("midi", midi_filename)


def cut_midi_in_measures(input_file, n_bins=4, n_parts=10):
    """
    Cuts midi files by n_bins partitions.
    """
    midifile = open_midi(input_file)
    for i in range(n_parts):
        try:
            midi_temp = m21.stream.Score()
            midi_temp.append(midifile.measures(i * n_bins, i * n_bins + n_bins))
            print(input_file[:-4] + "_" + str(i) + ".mid")
            midi_temp.write("midi", input_file[:-4] + "_" + str(i) + ".mid")
        except:
            break


# Methods actually being used


def notes_to_chords(input_file, file_name, tracks=[1, 2, 3, 4, 5], chords="major"):
    """
    Changes single notes in midi to chords.
    input_file  - path to the input midi file
    output_file - path to the output file to save
    tracks - list of tracks numers to modify
    chords - "major" or "minor" for corresponding triads
    """
    input_midi = MidiFile(input_file)
    output_midi = MidiFile()

    # Copying the time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    for track_n, original_track in enumerate(input_midi.tracks):
        # Loop through tracks in MIDI file

        if track_n in tracks:
            # TODO this will strip away all track
            # Not sure about this one I think I fixed it
            # Make sure that we only modify wanted midi track

            new_track = MidiTrack()

            for idx, msg in enumerate(original_track):
                # Loop through each message in original track
                # Good to know:
                # note_on & note_off represent veolocity.
                #   (A note_on message with zero velocity will also be counted as note_off)

                if msg.type in ["note_on", "note_off"]:
                    # Only try to make chords of our note messages
                    note = msg.note
                    if chords == "major":
                        new_track.append(msg.copy())
                        new_track.append(msg.copy(note=note + 4, time=0))
                        new_track.append(msg.copy(note=note + 7, time=0))
                        new_track.append(msg)
                        new_track.append(msg.copy(note=note + 4, time=0))
                        new_track.append(msg.copy(note=note + 7, time=0))
                    elif chords == "minor":
                        new_track.append(msg.copy())
                        new_track.append(msg.copy(note=note + 3, time=0))
                        new_track.append(msg.copy(note=note + 7, time=0))
                        new_track.append(msg)
                        new_track.append(msg.copy(note=note + 3, time=0))
                        new_track.append(msg.copy(note=note + 7, time=0))
                else:
                    new_track.append(msg)
            output_midi.tracks.append(new_track)
        else:
            output_midi.tracks.append(original_track)
    output_file_name, output_file_path, *_ = get_file_name_for_saving(
        "mid", file_name, "NOTES_TO_CHORDS"
    )
    output_midi.save(output_file_path)
    return output_file_path



def toggle_drums(input_file, file_name, drums_on=True):
    if drums_on:
        change_instruments(input_file, file_name, {"track_1": -1})
    else:
        change_instruments(input_file, file_name, {"track_1": 0})

def change_instruments(input_file, file_name, new_instruments_by_track):
    """
    Changes every instrument
    input_file - File to midi file
    output_file - File for new midi file
    new_instruments_by_track - Structure describing which track to be replaced and with what program { track_id: program_number }
    See program number here https://en.wikipedia.org/wiki/General_MIDI
    """
    mid = MidiFile(input_file)

    out = MidiFile()
    out.ticks_per_beat = mid.ticks_per_beat

    tracks = {
        1: "track_1",
        2: "track_2",
        3: "track_3",
        4: "track_4",
        5: "track_5"
    }

    for track_i, track in enumerate(mid.tracks):
        # Loop through each track

        print("track", track.name)

        if track_i not in tracks:
            print("Appending track as is and skipping: track_", track_i)
            out.tracks.append(track)
            continue

        key = tracks[track_i]
        if key not in new_instruments_by_track:
            out.tracks.append(track)
        
        new_track = MidiTrack()

        if new_instruments_by_track[key] is -1:
            print("Removing track by not appending", key)
            continue

        for message in track:
            # Loop through each message in track
            # Modify with program_change message and append or just append

            if message.type == "program_change":
                print("Modifying track")
                new_track.append(
                    message.copy(program=new_instruments_by_track[key])
                )

            else:
                new_track.append(message)
        out.tracks.append(new_track)
    output_file_name, output_file_path, *_ = get_file_name_for_saving(
        "mid", file_name, "CHANGE_INSTRUMENTS"
    )
    out.save(output_file_path)

    return output_file_path


def convert_midi_to_wav(input_file, file_name, soundfont=""):
    """
    Converts midi file from input_file to wav output_file with particulary soundfont.
    You have to install fluidsynth utility first. The soundfont waits for
    a whole path to sf2 file.
    """
    fs = FluidSynth(soundfont)
    output_file_name, output_file_path, *_ = get_file_name_for_saving(
        "wav", file_name, "CONVERT_MID_TO_WAV"
    )
    fs.midi_to_audio(input_file, output_file_path)
    return output_file_path


# These will be used but I've not checked them if they work yet
def tone_invert(input_file, output_file, basenote=50):
    """
    Example: tone_invert("/content/Ievan_Polkka.mid", "inverted.mid", (max_note + min_note)/2)
    Inverts tones over basenote and saves to output_file.
    Optimal basenote is (max_note + min_note)/2
    """
    # TODO This is ai4 / "tonal inversion"
    try:
        mid = MidiFile(input_file)
    except:
        return None
    inverted = MidiFile()

    for track in mid.tracks:
        new_track = MidiTrack()
        if "drum" in track.name.lower():
            new_track = track
        else:
            for message in track:
                if isinstance(message, MetaMessage):
                    new_track.append(message)
                else:
                    if "note" in dir(message):
                        inverted_note = basenote - (message.note - basenote)
                        new_track.append(
                            message.copy(note=inverted_note, time=int(message.time))
                        )
                    else:
                        new_track.append(message)

        inverted.tracks.append(new_track)
    try:
        inverted.save(output_file)
    except:
        pass


def invert_midi(input_file, output_file):
    """
    Example: invert_midi("/content/scooter-let_me_be_your_valentine.mid", "inverted.mid")
    Takes input_file, inverts its playback and saves to output_file
    """
    # TODO This is ai3 / reverse playback
    output_midi = MidiFile()
    tempo = 600000
    new_tracks = []
    try:
        input_midi = MidiFile(input_file)
    except:
        return None
    # Copying the time metrics between both files

    output_midi.ticks_per_beat = input_midi.ticks_per_beat
    for original_track in input_midi.tracks:
        for msg in original_track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

        for track_n, original_track in enumerate(input_midi.tracks):
            new_tracks.append(MidiTrack())
            for msg in original_track:
                new_tracks[track_n].append(msg)
    for track in new_tracks:
        output_midi.tracks.append(track[::-1])

    try:
        output_midi.save(output_file)
    except:
        pass
