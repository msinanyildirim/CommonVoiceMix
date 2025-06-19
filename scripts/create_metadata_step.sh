lang1="english"
lang2="german"

DATASET_ROOT=$(pwd)
metadata_outdir=${DATASET_ROOT}/cvmix_${lang1}_${lang2}/metadata

echo "Starting for ${lang1} and ${lang2}"
python -u scripts/create_cvmix_metadata.py --cv_dir ${DATASET_ROOT} \
                                --lang1 cv_${lang1} \
                                --lang2 cv_${lang2} \
                                --metadata_outdir ${metadata_outdir} 
