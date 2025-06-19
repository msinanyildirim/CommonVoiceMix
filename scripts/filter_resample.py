import argparse
import glob
import os

import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True,)
parser.add_argument("--lang", type=str, required=True, choices=['en', 'de'],)
parser.add_argument("--subset", type=str, required=True, choices=['train', 'dev', 'test'],)
parser.add_argument("--min_dur", type=int, default=7)

RATE = 16000

def resample(input_tuple):
        clip_path, target_path = input_tuple
        source, sr = sf.read(clip_path, dtype='float32')
        resampled = resample_poly(source, RATE, sr)
        sf.write(target_path, resampled, RATE)


def main(args):
    global lang
    lang = args.lang
    subset = args.subset
    global language
    language = "english" if lang == "en" else "german"
    data_folder = args.data_folder

    assert os.path.exists(os.path.join(data_folder, "cv-corpus-20.0-2024-12-06")), "cv-corpus-20.0-2024-12-06 folder could not be found"
    cv_folder = os.path.join(data_folder, "cv-corpus-20.0-2024-12-06")
    tsv_file = f"{cv_folder}/{lang}/{subset}.tsv"
    subset_df = pd.read_csv(tsv_file, sep="\t", usecols=["client_id", "path"])
    clip_durs_df = pd.read_csv(f"{cv_folder}/{lang}/clip_durations.tsv", sep="\t")

    os.makedirs(f"{data_folder}/cv_{language}", exist_ok=True)
    os.makedirs(f"{data_folder}/cv_{language}/clips", exist_ok=True)
    os.makedirs(f"{data_folder}/cv_{language}/metadata", exist_ok=True)

    clips_filtered = subset_df[subset_df["path"].isin(clip_durs_df[clip_durs_df["duration[ms]"] >= args.min_dur*1000]["clip"])]
    clips_to_save = clips_filtered.rename(columns={"path": "origin_path", "client_id": "speaker_ID"})
    clips_to_save["origin_path"] = "clips/" + clips_to_save["origin_path"].str.split(".").str[0] + ".wav"
    clips_to_save.to_csv(f"{data_folder}/cv_{language}/metadata/{subset}.csv", index=False)

    filtered_data = []
    to_resample = []
    for index, row in clips_filtered.iterrows():
        clip_name = row["path"]
        clip_path = f"{cv_folder}/{lang}/clips/{clip_name}"
        to_resample.append((clip_path, f"{data_folder}/cv_{language}/clips/{clip_name.split('.')[0]}.wav"))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(resample, to_resample)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"{args=}")
    main(args)
