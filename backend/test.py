import pickle
import numpy as np

index_to_word_path = '/Users/snehashislenka/Documents/NEU Assignments/Neural Network/Asg 5/asg-05-webapp/backend/index_to_word_stories_from_tagore.pkl'

word_to_index_path = '/Users/snehashislenka/Documents/NEU Assignments/Neural Network/Asg 5/asg-05-webapp/backend/word_to_index_stories_from_tagore.pkl'

index_to_word = []
word_to_index = []

with open(index_to_word_path, 'rb') as f:
    index_to_word = pickle.load(f)

with open(word_to_index_path, 'rb') as f:
    word_to_index = pickle.load(f)

# RNN model class (simplified for this example)
class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = None
        self.W = None
        self.V = None
        self.G = None

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

def forward_propagation(self, x):
    T = len(x)

    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)

    o = np.zeros((T, self.word_dim))

    for t in np.arange(T):
        e_t = self.G[x[t]]

        s[t] = np.tanh(self.U.dot(e_t) + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))

    return [o, s]

RNN.forward_propagation = forward_propagation

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = RNN(checkpoint['model_state']['V'].shape[0], hidden_dim=checkpoint['model_state']['U'].shape[0])
    model.U = checkpoint['model_state']['U']
    model.V = checkpoint['model_state']['V']
    model.W = checkpoint['model_state']['W']
    model.G = checkpoint['model_state']['G']
    
    # print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
    return model, checkpoint['epoch'], checkpoint['num_examples_seen'], checkpoint['learning_rate']

# Initialize model
vocabulary_size = 5242
model, checkpoint_epoch, checkpoint_num_examples_seen, checkpoint_learning_rate =  load_checkpoint('/Users/snehashislenka/Documents/NEU Assignments/Neural Network/Asg 5/asg-05-webapp/backend/checkpoint_epoch_41_2024-10-21 02:32:55.pkl')
# print(checkpoint_epoch, checkpoint_num_examples_seen, checkpoint_learning_rate)

sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
unknown_token = "UNKNOWN_TOKEN"

###########################################################
# generate new sequences based on the input feed sequence #
###########################################################

def generate_sequence(model, seed_sequence, num_words=80):
    # Convert the seed sequence to word indices
    input_sequence = [word_to_index.get(word, word_to_index[unknown_token]) for word in seed_sequence.split()]
    
    generated_sequence = input_sequence.copy()
    
    while num_words > 0:
        # Convert the generated sequence to integers
        int_sequence = [int(idx) for idx in generated_sequence]
        next_word_probs = model.forward_propagation(int_sequence)[0][-1]
        sampled_word = word_to_index[unknown_token]
        
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs)
            sampled_word = np.argmax(samples)
        
        if sampled_word == word_to_index[sentence_end_token]:
            continue

        generated_sequence.append(sampled_word)
        num_words -= 1
    
    # Convert word indices back to words, replacing sentence_end_token with a period
    generated_words = [index_to_word[int(idx)] if idx != word_to_index[sentence_end_token] else '.' for idx in generated_sequence]
    
    return generated_words

seed_text = "Today is a beautiful day"
generated_words = generate_sequence(model, seed_text, num_words=150)
print("Seed:", seed_text)
print("Generated:", " ".join(generated_words))