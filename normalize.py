import pandas as pd
from feats import FeatureExtract
import sys

folder = sys.argv[1]
for i in range(9):
  data = pd.read_csv(f"{folder}/sintoma_{i}.csv")
  X = data['texto']
  feats = FeatureExtract()
  normalized = pd.Series(feats.normalise_data(X))
  data['texto'] = normalized
  data.to_csv(f"{folder}/sintoma_{i}_normalized.csv")

