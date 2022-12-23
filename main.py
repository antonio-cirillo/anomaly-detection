from multiclass_classification.test.test_on_training_test import test_on_training_test
from multiclass_classification.test.test_with_pca import test_with_pca
from multiclass_classification.test.test_with_augmentation import test_with_augmentation

import os

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

test_on_training_test(DATASET_PATH)
test_with_pca(DATASET_PATH)
test_with_augmentation(DATASET_PATH)
