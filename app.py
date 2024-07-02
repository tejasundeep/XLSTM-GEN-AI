import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, sLSTMBlockConfig
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
text_data = """
Your sample text data goes here. Add more sentences to create a larger dataset.
For example, you can add some lines of poetry or prose.
"""

# Text Preprocessing
text_data = text_data.strip().replace('.', ' stopseq startseq ').replace('\n', ' ')
text_data = 'startseq ' + text_data + ' stopseq'

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text_data.split('startseq'):
    line = line.strip()
    if line:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = F.one_hot(torch.tensor(labels), num_classes=total_words).float()

# Define XLSTM Model Configuration
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4  # Example parameters, customize as needed
    ),
    slstm_block=sLSTMBlockConfig(
        backend="cuda", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"
    ),
    context_length=max_sequence_len-1,
    num_blocks=7,
    embedding_dim=100,
    slstm_at=[1],
)

# Initialize XLSTM Stack
xlstm_stack = xLSTMBlockStack(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlstm_stack = xlstm_stack.to(device)

# Define the full model
class XLSTMModel(nn.Module):
    def __init__(self, xlstm_stack, total_words):
        super(XLSTMModel, self).__init__()
        self.embedding = nn.Embedding(total_words, 100)
        self.xlstm_stack = xlstm_stack
        self.dense = nn.Linear(100, total_words)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = self.dense(x)
        x = self.log_softmax(x)
        return x

model = XLSTMModel(xlstm_stack, total_words)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
xs_tensor = torch.tensor(xs, dtype=torch.long).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(xs_tensor)
    loss = criterion(outputs.view(-1, total_words), labels_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Generate Text Function
def generate_text(seed_text, next_words, max_sequence_len):
    seed_text = 'startseq ' + seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        token_list = torch.tensor(token_list, dtype=torch.long).to(device)
        with torch.no_grad():
            predicted = model(token_list).cpu().numpy()
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, '')
        if output_word in ['stopseq', '']:
            break
        seed_text += " " + output_word
    return seed_text.replace('startseq ', '').replace(' stopseq', '')

seed_text = "Your starting seed text"
generated_text = generate_text(seed_text, 20, max_sequence_len)
print(generated_text)
