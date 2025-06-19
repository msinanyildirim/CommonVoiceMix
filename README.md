# CommonVoiceMix

This repository contains the scripts to recreate the CommonVoiceMix dataset used in [our paper](https://addlinkhere). The python code from [LibriMix](https://github.com/JorisCos/LibriMix) is adapted to work for CommonVoice dataset and creating language mixtures. Here are the steps to recreate our dataset:

1. Clone the repository and cd into it

```bash
git clone https://github.com/msinanyildirim/CommonVoiceMix.git
cd CommonVoiceMix
```

2. Download the raw data from CommonVoice

For our case, we use English and German as the languages and version 20 is the latest version. The code hardcodes this. If you want to use other langauges or later versions with more utterance, edit the code accordingly. 

Go to [CommonVoice](https://commonvoice.mozilla.org/en/datasets). Make sure English is chosen as the language and download Common Voice Corpus 20.0 version. Change the language to German and download the Common Voice Corpus 20.0 version. Both of them should be .tar.gz files. Put both in the CommonVoiceMix folder created and cd'ed into in the previous step.

3. Run the script

```bash
bash ./commonvoicemix.sh
```
This will first unzip both tar files in the current directory. Then, it filters samples longer than 7 seconds and resamples them to 16 kHz and collects them under their own folder for each language, cv_english and cv_german respectively. Then it uses the metadata inside the cvmix_english_german folder to recreate the mixtures. The resulting mixtures direcotry will have the same structure as the LibriMix.

If you want to use other langauge combinations, you can download the data for those languages. Also check the code in `scripts/create_metadata_step.sh`. You can disect the code in `commonvoicemix.sh`. The metadata should be created before the very last step of creating the mixtures.

If you use our code or the dataset, please consider citing our work.
```
Add paper info
```
