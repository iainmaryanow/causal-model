import causal_model
from numpy.random import normal
import pandas as pd

# SIZE = 2000
# A = normal(size=SIZE)
# B = A + normal(size=SIZE)
# C = A + normal(size=SIZE)
# D = B + C + normal(size=SIZE)
# E = D + normal(size=SIZE)

samples = 1000
A = normal(loc=10, size=samples)
B = normal(size=samples)
C = A + normal(size=samples)
D = B + C + normal(size=samples)

data = pd.DataFrame({
  'A': A,
  'B': B,
  'C': C,
  'D': D,
  # 'E': E
})
data = data.values.tolist()

if __name__ == '__main__':
  graph, marked = causal_model.build_causal_graph(data)
  is_graph_feasible = causal_model.is_feasible_causal_graph(graph, data)
  print(is_graph_feasible)