from multiclass_classification.test.test_on_training_test import test_on_training_test
from multiclass_classification.test.test_with_pca import test_with_pca
from multiclass_classification.test.test_different_weights import test_different_weights

import os

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
WEIGHTS = [0.25, 0.5, 0.75, 1]

test_on_training_test(DATASET_PATH)
test_with_pca(DATASET_PATH)
