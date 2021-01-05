import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_path ='wine_dataset.csv'
target='quality'

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv(data_path, sep=',')
tpot_data['type'] = tpot_data['type'].map({'white':0, 'red':1})
features = tpot_data.drop(target, axis=1)

tpot_data=tpot_data.apply(lambda x: x.fillna(x.mean()),axis=0)



#tpot_data = tpot_data.astype(np.float64)
print(tpot_data)


training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['quality'], random_state=667)

# Average CV score on the training set was: 0.9318147687570321
exported_pipeline = KNeighborsClassifier(n_neighbors=20, p=2, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
