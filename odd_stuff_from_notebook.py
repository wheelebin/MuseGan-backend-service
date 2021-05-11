def shuffle_midi(in_file, out_file):
    # This is the same as the other shuffle but modified
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
    num_measures = max([len(part) for part in midi_measures])
    measure_sample = random.sample(range(num_measures), num_measures)
    for part in midi_measures:
        partnew = m21.stream.Part()
        part_lst = list(part)
        for idx in measure_sample:
            try:
                partnew.append(part_lst[idx])
            except IndexError:
                continue
        # sample_part = random.sample(list(part), len(part))
        # for el in sample_part:
        #  partnew.append(el)
        s_out.append(partnew)

    s_out.write("midi", out_file)


def rename_files(input_folder):
    # This is never used
    file_list = []
    nums = []
    pattern = r".*kit(\d+).sf2$"
    for filename in glob(input_folder + "*.SF2") + glob(input_folder + "*.sf2"):
        mtch = re.search(pattern, filename.lower())
        if mtch:
            nums.append(int(mtch.group(1)))
        else:
            file_list.append(filename)
    idx = max(nums) + 1
    for filename in file_list:
        os.rename(filename, input_folder + f"kit{idx}.sf2")
        idx += 1
