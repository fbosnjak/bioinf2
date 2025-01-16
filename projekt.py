import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator



file_path = 'data.xls'

train_data = pd.read_excel(file_path, sheet_name='MILE')
test_data = pd.read_excel(file_path, sheet_name='BCCA')

train_data['Disease'] = train_data['Disease'].apply(lambda x: 1 if x == 'AML' else 0)
test_data['Disease'] = test_data['Disease'].apply(lambda x: 1 if x == 'AML' else 0)

train_data = train_data.drop(columns='Sample')
test_data = test_data.drop(columns='Sample')

train_data = train_data.rename(columns={'Disease': 'Effect'})
test_data = test_data.rename(columns={'Disease': 'Effect'})




model = BayesianNetwork()


nodes = ['Effect', 'ME1', 'ME2', 'ME4', 'ME5', 'ME6', 'ME7', 'ME8', 'ME9', 'ME10',
          'ME11', 'ME12', 'ME13', 'ME14', 'ME15', 'ME16', 'ME17', 'ME18', 'ME19', 'ME20',
        'ME21', 'ME22', 'ME23', 'ME24', 'ME25', 'ME26', 'ME27', 'ME28', 'ME29', 'ME30', 'ME31', 'ME32', 'ME33']

model.add_nodes_from(nodes = nodes)

edges = [('ME9', 'ME19'), ('ME9', 'ME2'), ('ME9', 'ME27'),('ME9', 'ME22'), ('ME9', 'ME30'), ('ME9', 'ME11'),
          ('ME2', 'ME10'), ('ME2', 'ME8'), ('ME2', 'ME19'), ('ME2', 'ME5'), ('ME2', 'ME20'), ('ME2', 'ME11'),
          ('ME5', 'ME19'), ('ME5', 'ME23'), ('ME5', 'ME27'),
          ('ME8', 'ME6'), ('ME8', 'ME15'), ('ME8', 'ME16'),
          ('ME23', 'ME29'), ('ME23', 'ME24'), ('ME23', 'ME27'), 
          ('ME22', 'ME27'), ('ME22', 'ME13'), ('ME22', 'ME26'), ('ME22', 'ME30'),
          ('ME17', 'ME25'), ('ME17', 'ME21'), ('ME17', 'ME3'), ('ME17', 'ME4'),
          ('ME6', 'Effect'), ('ME6', 'ME1'), ('ME6', 'ME31'), ('ME6', 'ME15'), 
          ('ME24', 'ME29'), ('ME24', 'ME16'), ('ME24', 'ME11'), 
          ('ME30', 'ME26'),
          ('ME3', 'ME21'), ('ME3', 'ME7'), ('ME3', 'ME25'), ('ME3', 'ME12'), ('ME3', 'ME10'),
          ('ME1', 'ME12'), ('ME1', 'ME4'), ('ME1', 'ME18'), 
          ('ME29', 'ME10'), ('ME29', 'ME19'), ('ME29', 'ME4'),
          ('ME16', 'ME19'), ('ME16', 'ME20'), 
          ('ME21', 'ME7'),
          ('ME12', 'Effect'), ('ME12', 'ME28'), ('ME12', 'ME14'), 
          ('ME10', 'ME7'), ('ME10', 'ME20'), ('ME10', 'ME13'), 
          ('ME4', 'ME14'), 
          ('ME13', 'ME25'), ('ME13', 'ME11'), 
          ('ME14', 'ME33'), ('ME14', 'ME32'), 
          ('ME31', 'ME32'), 
          ('ME11', 'ME18'),
          ('ME18', 'ME26') ]

model.add_edges_from(edges)

model.fit(train_data, estimator=MaximumLikelihoodEstimator)
