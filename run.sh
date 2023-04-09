export PYTHONPATH=.:$PYTHONPATH
export BUCKET=meliad_eu2
export EXP_NAME="v12_br"
# export EXP_NAME="cosine_pg19_4"

# gsutil cp gs://$BUCKET/experiments/$EXP_NAME/config.gin .
/usr/bin/env python3 transformer/ht_main_inference.py --alsologtostderr \
--gin_file config.gin \
--split=test \
--num_steps=100 \
--gin_param DecoderOnlyLanguageModel.output_token_losses=True \
--load_dir gs://$BUCKET/experiments/$EXP_NAME