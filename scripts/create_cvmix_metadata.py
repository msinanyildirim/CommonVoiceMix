import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

# A random seed is used for reproducibility
random.seed(72)

# A random seed is used for reproducibility
parser = argparse.ArgumentParser()
parser.add_argument("--cv_dir", type=str, required=True,
                    help="Path to the MLSMix1 root directory")
parser.add_argument("--lang1", type=str, required=True,
                    help="First language included in the mixtures")
parser.add_argument("--lang2", type=str, required=True,
                    help="Second language included in the mixtures")
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help='Where librimix metadata files will be stored.')

def main(args):
    lang1_dir = os.path.join(args.cv_dir, args.lang1)
    lang2_dir = os.path.join(args.cv_dir, args.lang2)
    # Create Librimix metadata directory
    md_outdir = args.metadata_outdir
    os.makedirs(md_outdir, exist_ok=True)
    create_librimix_metadata(lang1_dir, lang2_dir,
                             md_outdir)
    
def create_librimix_metadata(lang1_dir, lang2_dir,
                             md_outdir):
    # Dataset name
    dataset = f'cvmix'
    # Metadata files
    lang1_md_dir = os.path.join(lang1_dir, "metadata")
    lang2_md_dir = os.path.join(lang2_dir, "metadata")

    to_process = ["train.csv", "dev.csv", "test.csv"]

    to_be_ignored = []
    check_already_generated(md_outdir, dataset, to_be_ignored, to_process)

    print(f"{to_process=}", flush=True)
    for md_file in to_process:
        cvmix_md_file = f"{md_file}"
        print(f"Starting working on {cvmix_md_file}", flush=True)

        # Open original languages metadata files
        lang1_md = pd.read_csv(os.path.join(lang1_md_dir, cvmix_md_file), engine="python")
        lang2_md = pd.read_csv(os.path.join(lang2_md_dir, cvmix_md_file), engine="python")

        # Save path
        save_path = os.path.join(md_outdir,
                                 '_'.join([dataset, md_file]))
        
        print(f"Creating {os.path.basename(save_path)} file in {md_outdir}")
        subset = md_file.replace('.csv', '')

        mixtures_md = create_librimix_df(
            lang1_md, lang1_dir, lang2_md, lang2_dir, subset)
        
        # Round number of files
        mixtures_md = mixtures_md[:len(mixtures_md) // 100 * 100]
        

        # Save csv files
        mixtures_md.to_csv(save_path, index=False)


def check_already_generated(md_outdir, dataset, to_be_ignored, to_process):
    # Check if the metadata files in MLSMix1 already have been used
    already_generated = os.listdir(md_outdir)
    for generated in already_generated:
        if generated.startswith(f"{dataset}") and 'info' not in generated:
            if 'train' in generated:
                to_be_ignored.append('train.csv')
            elif 'dev' in generated:
                to_be_ignored.append('dev.csv')
            elif 'test' in generated:
                to_be_ignored.append('test.csv')
            print(f"{generated} already exists in "
                  f"{md_outdir} it won't be overwritten")
    for element in to_be_ignored:
        to_process.remove(element)


def create_librimix_df(lang1_md, lang1_dir, lang2_md, lang2_dir, subset):
    # Create a dataframe that will be used to generate sources and mixtures
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    # Add columns (depends on the number of sources)
    for i in range(2):
        mixtures_md[f"source_{i + 1}_path"] = {}
        mixtures_md[f"source_{i + 1}_gain"] = {}

    pairs = set_pairs(lang1_md, lang2_md)
    clip_counter = 0

    for pair in tqdm(pairs, total=len(pairs)):
        sources_info, sources_list_max = read_sources(
            lang1_md, lang2_md, lang1_dir, lang2_dir, pair, subset)
        
        # compute initial loudness, randomize loudness and normalize sources
        loudness, _, sources_list_norm = set_loudness(sources_list_max)
        # Do the mixture
        mixture_max = mix(sources_list_norm)
        # Check the mixture for clipping and renormalize if necessary
        renormalize_loudness, did_clip = check_for_cliping(mixture_max,
                                                           sources_list_norm)
        clip_counter += int(did_clip)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        row_mixture = get_row(sources_info, gain_list)
        mixtures_md.loc[len(mixtures_md)] = row_mixture

    print(f"Among {len(mixtures_md)} mixtures, {clip_counter} clipped.")
    return mixtures_md



def set_pairs(lang1_md, lang2_md):
    """ set pairs of sources to make the mixture """
    # Initialize list for pairs sources
    utt_pairs = []

    lang1_indices = list(range(len(lang1_md)))
    lang2_indices = list(range(len(lang2_md)))

    curr_index = 0
    while len(lang1_indices) > 0 and len(lang2_indices) > 0 and curr_index < 30000:
        lang1_index = random.sample(lang1_indices, 1)[0]
        lang2_index = random.sample(lang2_indices, 1)[0]

        couple = [lang1_index, lang2_index]
        utt_pairs.append(couple)

        lang1_indices.remove(lang1_index)
        lang2_indices.remove(lang2_index)

        curr_index += 1
    
    print(f"pair list length {len(utt_pairs)=}")
    return utt_pairs




def read_sources(lang1_md, lang2_md, lang1_dir, lang2_dir, pair, subset):
    source1 = lang1_md.iloc[pair[0]]
    source2 = lang2_md.iloc[pair[1]]

    if "mixture_ID" in source1:
        id_l = [source1["mixture_ID"], source2["mixture_ID"]]
        absolute_path1 = os.path.join(lang1_dir, "max", subset, "mix_clean", f"{id_l[0]}.wav")
        absolute_path2 = os.path.join(lang2_dir, "max", subset, "mix_clean", f"{id_l[1]}.wav")
    else:
        id_l = [os.path.split(source['origin_path'])[1].split('.')[0] for source in [source1, source2]]
        absolute_path1 = os.path.join(lang1_dir, subset, 'audio', id_l[0].split('_')[0], id_l[0].split('_')[1] , f"{id_l[0]}.flac")
        absolute_path2 = os.path.join(lang2_dir, subset, 'audio', id_l[1].split('_')[0], id_l[1].split('_')[1] , f"{id_l[1]}.flac")
        if not os.path.exists(absolute_path1):
            absolute_path1 = os.path.join(lang1_dir, f"{source1['origin_path'].split('.')[0]}.wav")
            absolute_path2 = os.path.join(lang2_dir, f"{source2['origin_path'].split('.')[0]}.wav")
            if not os.path.exists(absolute_path1):
                assert False, f"File {absolute_path1} does not exist"
    
    
    mixtures_id = "_".join(id_l)



    path_list = [source['origin_path'] for source in [source1, source2]]

    s1, _ = sf.read(absolute_path1, dtype='float32')
    s2, _ = sf.read(absolute_path2, dtype='float32')

    length_list = [len(s1), len(s2)]
    max_length = max(length_list)

    sources_list = [
        np.pad(s1, (0, max_length - len(s1)), mode='constant'),
        np.pad(s2, (0, max_length - len(s2)), mode='constant')
    ]

    sources_info = {'mixtures_id': mixtures_id, "path_list": path_list}

    return sources_info, sources_list


def set_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        # if i == len(sources_list) - 1:
        #     target_loudness = random.uniform(MIN_LOUDNESS - 5,
        #                                      MAX_LOUDNESS - 5)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max


def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def get_row(sources_info, gain_list, n_src=2):
    """ Get new row for each mixture/info dataframe """
    row_mixture = [sources_info['mixtures_id']]
    for i in range(n_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
    return row_mixture



if __name__ == "__main__":
    args = parser.parse_args()
    print(f"{args=}")
    main(args)
