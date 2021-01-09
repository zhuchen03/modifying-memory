ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

mkdir -p 'bert'
cd bert
wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
unzip cased_L-12_H-768_A-12
rm cased_L-12_H-768_A-12.zip
cd cased_L-12_H-768_A-12
wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
tar -xzf bert-base-cased.tar.gz
rm bert-base-cased.tar.gz
rm bert_model*
cd ../../
