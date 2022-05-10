from scipy.stats import pearsonr
import itertools

ARROW_FLAG = 'arrow'

def build_causal_graph(data, independence_threshold=0.1):
  variables = list(range(len(data[0])))

  graph = []
  marked = []
  for i in variables:
    graph.append([])
    marked.append([])
    for j in variables:
      graph[i].append(True if i != j else False)
      marked[i].append(False)

  separating_sets = {}
  for N in range(len(variables)+1):
    for x, y in list(itertools.combinations(variables, 2)):
      x_neighbors = list(adjacent(x, graph))
      y_neighbors = list(adjacent(y, graph))
      z_candidates = list(set(x_neighbors + y_neighbors) - set([x, y]))

      for z in itertools.combinations(z_candidates, N):
        correlation = _compute_correlation([x, y], list(z), data)
        if not (x, y) in separating_sets and abs(correlation) <= independence_threshold:
          graph[x][y] = False
          graph[y][x] = False
          separating_sets[(x, y)] = z
          break

  for z in variables:
    for x, y in itertools.combinations(adjacent(z, graph), 2):
      if not graph[x][y] and not graph[y][x]:
        sep = separating_sets.get((x, y), None) or separating_sets.get((y, x), None)
        if not sep or z not in sep:
          graph[x][z] = 'arrow'
          graph[z][x] = False
          graph[y][z] = 'arrow'
          graph[z][y] = False

  added_arrows = True
  while added_arrows:
    R1_added_arrows = _apply_recursion_rule_1(graph, marked)
    R2_added_arrows = _apply_recursion_rule_2(graph, marked)
    added_arrows = R1_added_arrows or R2_added_arrows

  return graph, marked


def adjacent(x, graph):
  adjs = set()
  for i in range(len(graph)):
    if graph[i][x] != False:
      adjs.add(i)
    if graph[x][i] != False:
      adjs.add(i)

  return adjs


def edges(graph):
  edge_set = set()
  for i in range(len(graph)):
    for j in range(len(graph)):
      if graph[i][j] != False:
        edge_set.add(tuple(sorted((i, j))))

  return edge_set


def _apply_recursion_rule_1(graph, marked):
  added_arrows = False

  for c in range(len(graph[0])):
    for a, b in itertools.combinations(adjacent(c, graph), 2):
      if not graph[a][b]:
        if arrow(a, c, graph) and not arrow(b, c, graph) and not (arrow(c, b, graph) and marked[b][c]):
          graph[c][b] = ARROW_FLAG
          graph[b][c] = False
          marked[c][b] = True
          marked[b][c] = True
          added_arrows = True

        if arrow(b, c, graph) and not arrow(a, c, graph) and not (arrow(c, a, graph) and marked[a][c]):
          graph[c][a] = ARROW_FLAG
          graph[a][c] = False
          marked[c][a] = True
          marked[a][c] = True
          added_arrows = True

  return added_arrows


def _apply_recursion_rule_2(graph, marked):
  added_arrows = False
  for a, b in edges(graph):
    if graph[a][b] != ARROW_FLAG and _marked_directed_path(a, b, graph):
      graph[a][b] = ARROW_FLAG
      added_arrows = True

  return added_arrows


def arrow(src, dest, graph):
  return graph[src][dest] == ARROW_FLAG


def _marked_directed_path(a, b, graph):
  seen = [a]
  neighbors = [(a, neighbor) for neighbor in adjacent(a, graph)]

  while neighbors:
    parent, child = neighbors.pop()
    if graph[parent][child] == ARROW_FLAG and marked[parent][child]:
      if child == b:
        return True
      if child not in seen:
        neighbors += [(child, neighbor) for neighbor in adjacent(child, graph)]
      seen.append(child)

  return False


def is_feasible_causal_graph(causal_graph, data, independence_threshold=0.1):
  basis_set = _compute_basis_set(causal_graph)
  for statement in basis_set:
    if not _is_valid_d_separation_statement(statement[0], statement[1], data, independence_threshold):
      return False

  return True


# Find all non-adjacent pairs
# Get causal parents for each of the pairs
# Construct the d-separation statements
def _compute_basis_set(causal_graph):
  non_adjacent_pairs = _get_non_adjacent_pairs(causal_graph)
  causal_parents = _get_causal_parents(non_adjacent_pairs, causal_graph)
  return _build_d_separation_statements(non_adjacent_pairs, causal_parents)


# Return all pairs that neither A -> B or B -> A
def _get_non_adjacent_pairs(causal_graph):
  non_adjacent_pairs = []
  for i in range(len(causal_graph)):
    for j in range(i+1, len(causal_graph)):
      if not causal_graph[i][j] and not causal_graph[j][i]:
        non_adjacent_pairs.append((i, j))

  return non_adjacent_pairs


# Return list of all variables that point to either of the non-adjacent pairs
def _get_causal_parents(non_adjacent_pairs, causal_graph):
  causal_parents = []
  for variable1, variable2 in non_adjacent_pairs:
    parents = []
    for index in range(len(causal_graph)):
      if causal_graph[index][variable1] or causal_graph[index][variable2]:
        parents.append(index)
    causal_parents.append(parents)

  return causal_parents


# Concatenate the non-adjacent pairs with their parents to construct the d-separation statement
def _build_d_separation_statements(non_adjacent_pairs, causal_parents):
  d_separation_statements = []
  for index, pair in enumerate(non_adjacent_pairs):
    statement = (pair, causal_parents[index])
    d_separation_statements.append(statement)

  return d_separation_statements


# A d-separation statement is valid when the correlation is within
# the independence threshold (either positive or negative)
def _is_valid_d_separation_statement(variables, conditioned_variables, data, independence_threshold):
  correlation = _compute_correlation(variables, conditioned_variables, data)
  return abs(correlation) <= independence_threshold


# Recursively condition on variables until correlation can be computed
def _compute_correlation(variables, conditioned_variables, data):
  # Directly compute the correlation if there a no conditions
  if not len(conditioned_variables):
    return _calculate_pearson(variables, data)
  
  # Bin the first conditioned variable data (others will be recursed later)
  conditioned_variable = conditioned_variables[0]
  conditioned_data = set(map(lambda x: int(10 * x[conditioned_variable]), data))

  # Keep track of weights as not all data will be able to have correlation computed
  correlation = 0
  total_weights = 0

  for unique_value in conditioned_data:
    # Use only the data that has the specific condition for the variable
    data_subset = list(filter(lambda x: int(10 * x[conditioned_variable]) == unique_value, data))
    
    # Correlation only works with more than 2 values
    if len(data_subset) < 2:
      continue
    
    weight = len(data_subset) / len(data)
    total_weights += weight
    
    # Sum weighted correlations of recursed conditioned variables with the subsetted data
    correlation += weight * _compute_correlation(variables, conditioned_variables[1:], data_subset)

  # Return the weighted averaged correlation if it could be computed
  return 0 if total_weights == 0 else correlation / total_weights


def _calculate_pearson(variables, data):
  variable_1_data = list(map(lambda x: x[variables[0]], data))
  variable_2_data = list(map(lambda x: x[variables[1]], data))
  return pearsonr(variable_1_data, variable_2_data)[0]