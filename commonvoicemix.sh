#!/usr/bin/env bash

set -euo pipefail

# Find the full path of the current folder
DATASET_ROOT=$(pwd)
echo "Dataset root is ${DATASET_ROOT}"

# Untar the corpus
echo "Untarring the English corpus"
tar --checkpoint=10000 --checkpoint-action=dot -xzf cv-corpus-20.0-2024-12-06-en.tar.gz
printf "\n"
echo "Untarring the German corpus" 
tar --checkpoint=10000 --checkpoint-action=dot -xzf cv-corpus-20.0-2024-12-06-de.tar.gz
printf "\n"

# This script will filter out samples shorter than 7 seconds and resample them to 16 kHz and save them in a separate cv_language folder
for lang in en de; do
	for subset in train dev test; do
        echo "Filtering and resampling ${lang} ${subset}"
		python scripts/filter_resample.py --data_folder ${DATASET_ROOT} --lang ${lang} --subset ${subset}
	done;
done;

# This script uses the given metadata to create the mixtures
# If you want to create your own mixtures, you can use the scripts/create_cvmix_metadata.sh script before the next step
languages="english german"
metadata_dir=${DATASET_ROOT}/cvmix_$(echo "$languages" | tr ' ' '_')/metadata

echo "Starting creating mixtures for ${languages}"
python scripts/create_cvmix_from_metadata.py --cv_dir ${DATASET_ROOT} \
                                --languages ${languages} \
                                --metadata_dir ${metadata_dir}
