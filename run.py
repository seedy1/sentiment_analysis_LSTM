import numpy as np
import remove_punctuations
from string import punctuation
from collections import Counter
from model import SentimentModel
import joblib

import torch
from torch.utils.data import DataLoader, TensorDataset

#read only
with open('data/reviews.txt', 'r') as file:
    reviews = file.readlines()

with open('data/labels.txt', 'r') as file:
    labels = file.readlines()

# print(reviews, labels)
# reviews = remove_punctuations(reviews)

all_reviews = list()

for text in reviews:
    text = text.lower()
    text = ''.join([ char for char in text if char not in punctuation ])
    all_reviews.append(text)
all_text = ' '.join(all_reviews)
all_words = all_text.split()

#get word counts
count_words = Counter(all_words)
total_words = len(all_words)
top_words = count_words.most_common(total_words)
print(top_words[:20])

# custom word2Vec

# vocab_to_int = {}
# for i, (w,c) in enumerate(top_words):
#     vocab_to_int={w:i+1}
vocab_to_int={w:i+1 for i,(w,c) in enumerate(top_words)}
# print(vocab_to_int)

# encoded reviews to int
encoded_reviews = list()
for review in all_reviews:
    encoded_review = list()
    for word in review.split():
        if word not in vocab_to_int.keys():
            encoded_review.append(0)
        else:
            encoded_review.append(vocab_to_int[word])
    encoded_reviews.append(encoded_review)

# pad the sequences into the same length
# we do not need the whole text to know wether it was positive or negative
max_sequence_length = 250
features = np.zeros(( len(encoded_reviews), max_sequence_length ), dtype=int)

for i, review in enumerate(encoded_reviews):
    review_length = len(review)

    if review_length <= max_sequence_length:
        zeros = list(np.zeros(max_sequence_length-review_length))
        new = zeros+review
    else:
        new = review[:max_sequence_length]
    features[i,:] = np.array(new)

#one hot label the labels :-)
labels = [ 1 if label.strip() == 'positive' else 0 for label in labels ]

# train val split
x_train = features[:int(0.8*len(features))]
y_train = labels[:int(0.8*len(features))]
x_valid = features[int(0.8*len(features)):int(0.9*len(features))]
y_valid = labels[int(0.8*len(features)):int(0.9*len(features))]
x_test = features[int(0.9*len(features)):]
y_test = labels[int(0.9*len(features)):]
print(len(y_train), len(y_valid), len(y_test))

# create dataloader
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
valid_data = TensorDataset(torch.LongTensor(x_valid), torch.LongTensor(y_valid))
test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

model = SentimentModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
# print(model)

# loss and optimization functions
lr=0.01
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training params

epochs = 2
counter = 0
print_every = 100
clip = 5

model.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = model.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        # if(train_on_gpu):
        #     inputs=inputs.cuda()
        #     labels=labels.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output, h = model(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                # inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

# test
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = model.init_hidden(batch_size)

model.eval()
# iterate over test data
for inputs, labels in test_loader:

    h = tuple([each.data for each in h])



    output, h = model(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

print(' S A V I N G  M O D E L')

joblib.dump(model, 'model.pkl')

print('M O D E L  S A V E D..............')