
cp modeling_phi.py "$CONDA_PREFIX/lib/python3.9/site-packages/transformers/models/phi/"
mkdir -p "$HOME/nltk_data/corpora"
unzip wordnet.zip
cp -r wordnet "$HOME/nltk_data/corpora"

mkdir -p ./data/anno_data
mkdir -p ./data/modelnet40_data
mkdir -p ./data/objaverse_data