import random
# hint: these two imports might be helpful for your implementation
from collections import defaultdict
from nanogpt.utils import evaluate_places

# find the set of places and the most common place in the training set
############# YOUR CODE HERE #############
training_path = 'nanogpt/data/birth_places_train.tsv'
test_path = 'nanogpt/data/birth_places_test.tsv'
train_data = open(training_path).read()
train_data = train_data.split('\n')
place_counts = defaultdict(int)
for line in train_data[:-1]:
    x = line.split('?')
    place = x[1].strip()
    place_counts[place] += 1

places_set = list(set(place_counts.keys()))
most_common = max(place_counts, key=place_counts.get)

test_data = open(test_path).read()
test_data = test_data.split('\n')
N = len(test_data)-1
##########################################

# random baseline
############# YOUR CODE HERE #############
predicted_rand = [random.choice(places_set) for _ in range(N)]
total,correct = evaluate_places(test_path, predicted_rand)
##########################################
print(f'Random baseline accuracy: {correct} out of {total} ({correct/total*100}%)')

# most-common place baseline
############# YOUR CODE HERE #############
predicted_common = [most_common for _ in range(N)]
total,correct = evaluate_places(test_path, predicted_common)
##########################################
print(f'Most-common baseline accuracy: {correct} out of {total} ({correct/total*100}%)')