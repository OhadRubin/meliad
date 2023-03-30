export PYTHONPATH=.:$PYTHONPATH

# --gin_file=size/medium150M.gin \

set -e
on_error(){
    HOSTNAME=$(hostname)
  /home/ohadr/.local/bin/alerts msg --message "Error from $HOSTNAME, $WANDB_NAME"
  sleep 100000
}
 
trap 'on_error' ERR
# export DEBUG=1
# export BUCKET=meliad_eu2
#  ./bin/interactive_tpu.py --node_id 4 --version 4 --zone "us-central2-b" --bucket "meliad2_us2" --cores 64 --project "tpu-project-2-379909" --repo_name meliad
export BUCKET=meliad2_us2

# gsutil cp gs://$BUCKET/experiments/cosine_pg19/config.gin .
/usr/bin/env python3 transformer/ht_main.py --alsologtostderr \
--gin_file=base_htrans.gin \
--gin_file=options/seq_4096.gin \
--gin_file=options/positions_t5.gin \
--gin_file=options/lr_cosine_decay.gin \
--gin_file=tasks/pg19_tokens.gin \
--gin_file=recurrent/bias_skip.gin \
--gin_file=/home/ohadr/meliad/transformer/configs/size/medium_150M.gin \
--workdir=gs://$BUCKET/experiments/cosine_pg19_2

# /usr/bin/env python3 transformer/ht_main_inference.py --alsologtostderr \
# --gin_file=config.gin \
# --gin_param DecoderOnlyLanguageModel.output_token_losses=True \
# --workdir=gs://$BUCKET/experiments/cosine_pg19
# --gin_file=options/seq_4096.gin \
# --gin_file=options/positions_t5.gin \
# --gin_file=options/lr_cosine_decay.gin \
# --gin_file=tasks/pg19_tokens.gin \
# --gin_file=recurrent/bias_skip.gin \
# /usr/bin/env python3 transformer/ht_main_inference.py --workdir=gs://$BUCKET/experiments/cosine_pg19
# gsutil rsync -r gs://meliad2_us2/experiments/cosine_pg19 gs://meliad_eu2/experiments/cosine_pg19
# CMD="python3 transformer/ht_main.py --alsologtostderr --gin_file=base_htrans.gin --gin_file=options/seq_4096.gin --gin_file=options/positions_t5.gin --gin_file=options/lr_cosine_decay.gin --gin_file=tasks/pg19_tokens.gin --gin_file=recurrent/bias_skip.gin --workdir=gs://meliad2_us2/experiments/cosine_pg19"
# gcloud alpha compute tpus tpu-vm ssh v4-64-node-4 --project=tpu-project-2-379909 --zone=us-central2-b  --command="export PYTHONPATH=.:\$PYTHONPATH &&  cd /home/ohadr/meliad; $CMD"
