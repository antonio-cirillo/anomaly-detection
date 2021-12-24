from test.test_different_weights import test_different_weights
from test.test_with_augmentation import test_with_augmentation
from test.test_with_merging import test_with_merging
from test.test_binary_classification import test_binary_classification

WEIGHTS = [0.25, 0.5, 0.75, 1]

test_different_weights(WEIGHTS)

test_with_augmentation(WEIGHTS)

test_with_merging(WEIGHTS)

test_binary_classification()
