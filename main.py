import causal_model
from numpy.random import normal
import pandas as pd

SAMPLES = 1000

if __name__ == '__main__':
  # A = normal(size=SAMPLES)
  # B = A + normal(size=SAMPLES)
  # C = A + normal(size=SAMPLES)
  # D = B + C + normal(size=SAMPLES)
  # E = D + normal(size=SAMPLES)

  # data = pd.DataFrame({
  #   'A': A,
  #   'B': B,
  #   'C': C,
  #   'D': D,
  #   'E': E
  # })


  A = normal(loc=10, size=SAMPLES)
  B = normal(size=SAMPLES)
  C = A + normal(size=SAMPLES)
  D = B + C + normal(size=SAMPLES)

  data = pd.DataFrame({
    'A': A,
    'B': B,
    'C': C,
    'D': D
  })

  data = data.values.tolist()

  graph, marked = causal_model.build_causal_graph(data)
  is_graph_feasible = causal_model.is_feasible_causal_graph(graph, data)
  print(is_graph_feasible)