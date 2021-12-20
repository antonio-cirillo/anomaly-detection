from test.testDifferentsWeight import test_differents_weight
from test.testWithSmote import test_with_smote
from test.testWithRandomOverSampler import test_with_random_over_sampler

# Eseguo il primo test:
# Effettuiamo un primo test con i pesi w = [0.25, 0.5, 0.75, 1]
WEIGHTS = [0.25, 0.5, 0.75, 1]
test_differents_weight(WEIGHTS)

# Eseguo il secondo test:
# Effettuiamo il test precedente effettuando oversampling
test_with_smote(WEIGHTS)

# Eseguo il terzo test:
# Effettuiamo il primo test ignorando la labels relativi agli attacchi DoS
test_with_random_over_sampler(WEIGHTS)