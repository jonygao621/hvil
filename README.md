# Setup

```
pip install -r requirements.txt
```
  

# Usage

To generate data:
```
python data.py --env Acausal --agent Acausal --data config/data/acausal.json
```

To train a model:
```
python train.py --env Acausal --agent Acausal --train config/train/acausal.json
python train.py --env Acausal --agent RNN --train config/train/acausal.rnn.json
```
