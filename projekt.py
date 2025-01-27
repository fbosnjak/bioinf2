import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data_file = "data.xls"
mile_data = pd.read_excel(data_file, sheet_name='MILE')
bcca_data = pd.read_excel(data_file, sheet_name='BCCA')


# Step 2: Preprocess the data
def preprocess_data(df):
    
    # Remove rows where all columns except 'Sample' are empty or invalid
    df = df[~((df.drop(columns=['Sample'], errors='ignore').isna().all(axis=1)) | (df.drop(columns=['Sample'], errors='ignore') == "").all(axis=1))]
    
    # Remove 'Sample' column
    df = df.drop(columns=['Sample'], errors='ignore')
    
    # Transform 'Disease' column
    df['Disease'] = df['Disease'].map({'AML': 1, 'MDS': -1})
    
    # Rename 'ME0' to 'Effect'
    df = df.rename(columns={'ME0': 'Effect'})

    # Transform decimal values to 1 if positive, -1 if negative
    df = df.apply(lambda col: col.map(lambda x: 1 if x > 0 else -1 if x < 0 else x) if col.dtype != 'object' else col)
    
    return df

mile_data = preprocess_data(mile_data)
bcca_data = preprocess_data(bcca_data)

# Step 3: Define the Bayesian Network structure
nodes = ['Effect', 'ME1', 'ME2', 'ME4', 'ME5', 'ME6', 'ME7', 'ME8', 'ME9', 'ME10',
         'ME11', 'ME12', 'ME13', 'ME14', 'ME15', 'ME16', 'ME17', 'ME18', 'ME19', 'ME20',
         'ME21', 'ME22', 'ME23', 'ME24', 'ME25', 'ME26', 'ME27', 'ME28', 'ME29', 'ME30', 'ME31', 'ME32', 'ME33']

edges = [('ME9', 'ME19'), ('ME9', 'ME2'), ('ME9', 'ME27'), ('ME9', 'ME22'), ('ME9', 'ME30'), ('ME9', 'ME11'),
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
         ('ME18', 'ME26')]

# Create the Bayesian Network
model = BayesianNetwork(edges)

# Step 4: Train the Bayesian Network
mile_train_data = mile_data.drop(columns=['Disease'])
model.fit(mile_train_data, estimator=MaximumLikelihoodEstimator)

# Step 5: Test the Bayesian Network
def testNetwork(featureName):
    bcca_test_data = bcca_data.drop(columns=['Disease', featureName])
    predicted_effect = model.predict(bcca_test_data)[featureName]

    # Step 5.1: Evaluate the model
    if (featureName == 'Effect'):
        accuracy = accuracy_score(bcca_data['Disease'], predicted_effect)
        print(f"Accuracy of the Bayesian Network: {accuracy:.2f}")
    else:
        accuracy = accuracy_score(bcca_data[featureName], predicted_effect)
        print(f"Accuracy of the Bayesian Network for feature {featureName}: {accuracy:.2f}")

for feature in nodes:
    testNetwork(feature)


# Step 6: Compute ROC Curve
true_labels, predicted_labels = testNetwork("Effect")

# Convert -1,1 labels to 0,1 for ROC
true_labels = (true_labels + 1) // 2
predicted_labels = (predicted_labels + 1) // 2

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

# Step 7: Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


