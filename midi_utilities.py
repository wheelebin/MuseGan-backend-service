import os
from glob import glob
from struct import error
from mido import MidiFile, MidiTrack, tick2second, MetaMessage
from pypianoroll import Multitrack, outputs, track
import music21 as m21
import random
from midi2audio import FluidSynth
from utilities import get_file_name_for_saving
from config import DATA_DIR
import sqlite3
from datetime import datetime
import random


def cut_midi(input_midi, start_time, end_time):
    """
    Cuts midi input_file from start_time to end_time and saves cutted part into
    output_file.
    """
    print('INP: ', input_midi.length)
    output_midi = MidiFile()
    tempo = 600000
    if input_midi.type == 2:
        print("Can't cut the file")

    # Copying the time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    for original_track in input_midi.tracks:
        for msg in original_track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
    
    for original_track in output_midi.tracks:
        for msg in original_track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

    for original_track in input_midi.tracks:
        new_track = MidiTrack()
        time_passed = 0
        for msg in original_track:
            if msg.type in ["note_on", "note_off"]:
                time_passed += tick2second(msg.time, input_midi.ticks_per_beat, tempo)
                if time_passed > start_time and time_passed < end_time:
                    new_track.append(msg)
        output_midi.tracks.append(new_track)

    return output_midi


def join_midi(input_files, output_file=None):
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

    if output_file:
        output_midi.save(output_file)
    return output_midi

def stretch_midi_to_length(input_midi_path, file_name, length):
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
    for idx, file in enumerate([input_midi_path, input_midi_path]):
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

    output_file_name, output_file_path, *_ = get_file_name_for_saving(
        "mid", file_name, "STRETCH_MIDI"
    )
    output_midi.save(output_file_path)
    return output_file_path



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



def set_drums(input_file, file_name, drums_on=True):
    if drums_on:
        return change_instruments(input_file, file_name, {"track_1": 0})
    else:
        return change_instruments(input_file, file_name, {"track_1": -1})


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

        # Append index 0 in tracks, it's meta info
        if track_i not in tracks:
            out.tracks.append(track)
            continue

        key = tracks[track_i]

        # Append track as is if not in new_instruments_by_track arr
        if key not in new_instruments_by_track or new_instruments_by_track[key] == None:
            out.tracks.append(track)
            continue
        
        new_track = MidiTrack()

        # Don't append track if -1, will pretty much "delete" the track
        if new_instruments_by_track[key] == -1:
            continue

        for message in track:
            # Loop through each message in track
            # Modify with program_change message and append or just append

            if message.type == "program_change":
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

def tonal_inversion(input_file, basenote=None):

    if basenote == None:
        min_note, max_note = check_tones(input_file)
        basenote = (max_note + min_note)/2    
    
    return tone_invert(input_file, basenote)

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

def tone_invert_main(mid, basenote):
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
                        #TODO: Sometimes the inverted_note is not in between 0...127 (Like -1)
                        # Handle that
                        new_track.append(
                            message.copy(note=int(inverted_note), time=int(message.time))
                        )
                    else:
                        new_track.append(message)

        inverted.tracks.append(new_track)
    return inverted

# These will be used but I've not checked them if they work yet
def tone_invert(input_file, basenote=50):
    """
    Example: tone_invert("/content/Ievan_Polkka.mid", "inverted.mid", (max_note + min_note)/2)
    Inverts tones over basenote and saves to output_file.
    Optimal basenote is (max_note + min_note)/2
    """
    # TODO This is ai3 / "tonal inversion"
    mid = MidiFile(input_file)

    return tone_invert_main(mid, basenote)


def invert_midi_main(input_midi):
    output_midi = MidiFile()
    output_midi.ticks_per_beat = input_midi.ticks_per_beat
    
    new_tracks = []

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
    return output_midi

def invert_midi(input_file):
    """
    Example: invert_midi("/content/scooter-let_me_be_your_valentine.mid", "inverted.mid")
    Takes input_file, inverts its playback and saves to output_file
    """
    # TODO This is ai4 / reverse playback
    
    try:
        input_midi = MidiFile(input_file)
    except:
        return None

    return invert_midi_main(input_midi)

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'

def get_midi_by_genre(genre):
    genres_dir = DATA_DIR + '/01_Genres'
    output_dir = DATA_DIR + '/results/sorted'

    genre_folder = genres_dir + '/' + genre
    if os.path.isdir(genre_folder) == False:
        return None

    genre_folder = genres_dir + '/' + genre

    midi_files = []
    mid_files = glob(genre_folder + '/**/*.mid', recursive=True)
    MID_files = glob(genre_folder + '/**/*.MID', recursive=True)

    if mid_files:
        midi_files.extend(mid_files)

    if MID_files:
        midi_files.extend(MID_files)

    return random.choice(midi_files)

def open_db():
    con = sqlite3.connect('midi.sqlite3')
    cur = con.cursor()
    
    try:
        # Create table
        cur.execute('''CREATE TABLE conversions
                (date text, success boolean, originalFile text, newFile text, msg text)''')
    except:
        pass
    return con, cur

def insert_conversion_in_db(con, cur, success, originalFile, newFile, msg):
    #now = datetime.now()
    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    ## Fix bellow
    #command = "INSERT INTO conversions VALUES ('%s','%s','%s', '%s', '%s')" % (dt_string, success, originalFile.replace("'", "").replace('"', ""), newFile, msg)
    #cur.execute(command)
    #con.commit()
    return


#con, cur = open_db()


    

def check_midi(file):

    file_size_b = os.path.getsize(file)
    formated_size, label = format_bytes(file_size_b)

    if formated_size < 10 and label == 'kilobytes':
        os.remove(file)
        print("DELETED MIDI FILE: [REASON: %s] %s" % ('Too small', file))
        return None

    mid = MidiFile(file)
    # Double mid if shorter than 3 minutes (180 seconds)
    if mid.type != 2 and mid.length < 180:
        mid = join_midi([file, file])

    if mid.type != 2 and mid.length > 500:
        mid = cut_midi(mid, 0, 500)

    return True

def proccess_midi(file, file_name, output_dir, genre):

    mid = MidiFile(file)
    # Double mid if shorter than 3 minutes (180 seconds)
    if mid.length < 180:
        mid = join_midi([file, file])


    file_size_b = os.path.getsize(file)
    formated_size, label = format_bytes(file_size_b)

    if formated_size < 10 and label == 'kilobytes':
        os.remove(file)
        print("DELETED MIDI FILE: [REASON: %s] %s" % ('Too small', file))
        return None

    
    # Try rover on midi
    output_midi = invert_midi_main(mid)
    if output_midi is None or output_midi.length == 0:
        os.remove(file)
        print("DELETED MIDI FILE: [REASON: %s] %s" % ('Cant convert ROVER', file))
        return None


    
    # Try tonal inversion on file
    # I don't think I need to try tonal inversion on rover output as well
    output_midi = tone_invert_main(mid, 50)
    if output_midi is None or output_midi.length == 0:
        os.remove(file)
        print("DELETED MIDI FILE: [REASON: %s] %s" % ('Cant convert INVERSION', file))
        return None

    
    # Save original file since it passed above conditionals
    og_midi = MidiFile(file)
    output_file_name, *_ = get_file_name_for_saving(
        "mid", file_name, genre
    )
    output_file_path = "%s/%s.mid" % (output_dir, output_file_name)
    og_midi.save(output_file_path)
    return output_file_path



def proccess_raw_midis(midi_files, output_dir, genre):
    for file in midi_files:

        file_name, *_ = get_file_name_for_saving()

        
        #rows = cur.execute('SELECT * FROM conversions WHERE originalFile = "%s"' % file.replace("'", "").replace('"', ""))
        #for _ in rows: continue


        try:
            proccessed_output = proccess_midi(file, file_name, output_dir, genre)
            #if proccessed_output == None:
            #    insert_conversion_in_db(con, cur, False, file, '', 'DELETED')
        except:
            print("PROCCESSING FAILED, SKIPPING: ", file)
            #insert_conversion_in_db(con, cur, False, file, '', 'EXCEPTION')
            continue
        
        

        # Save original file since it passed above conditionals
        og_midi = MidiFile(file)
        output_file_name, *_ = get_file_name_for_saving(
            "mid", file_name, genre
        )
        output_file_path = "%s/%s.mid" % (output_dir, output_file_name)
        og_midi.save(output_file_path)

        #insert_conversion_in_db(con, cur, True, file, output_file_path, 'CONVERTED')
        

def proccess_raw_genres():
    genres_dir = DATA_DIR + '/01_Genres'
    output_dir = DATA_DIR + '/results/sorted'
    genre_folders = os.listdir(genres_dir)


    mid_files_amount = len(glob(genres_dir+'/**/*.mid'))
    MID_files_amount = len(glob(genres_dir+'/**/*.MID'))
    print('Amount of mid files to proccess: ', mid_files_amount + MID_files_amount)


    for genre in genre_folders:
        genre_output_dir = "%s/%s" % (output_dir, genre)
        if os.path.isdir(genre_output_dir) == False:
            os.mkdir(genre_output_dir)
            break

        genre_folder = genres_dir + '/' + genre

        midi_files = []
        mid_files = glob(genre_folder + '/**/*.mid', recursive=True)
        MID_files = glob(genre_folder + '/**/*.MID', recursive=True)

        if mid_files:
            midi_files.extend(mid_files)

        if MID_files:
            midi_files.extend(MID_files)

        print('Amount of mid files to proccess: ', len(midi_files))
        
        proccess_raw_midis(midi_files, genre_output_dir, genre)



#proccess_raw_genres()
#con.close()