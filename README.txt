Train LSTM on one of two tasks:
- Run the LSTM_keras script for binary classification
- Run the LSTM_categories script for numerical rating prediction

Running these scripts will process or load the necessary data
files, and build and train the LSTM network. The resulting
model weights, intermediate activations produced by the test
and training sets, class predictions, and prediction accuracy
is pickled and stored in the results folder.

Running the binary classification task will yield slightly
different accuracy percentages each time, even though the data
set can be reproducibly randomized with a seed. The randomized
selection of batches in the actual training of the model leads
to the accuracy variation across trials. The accuracy of the
trials was generally above 80%, though in some trials it was
as low as 66.5%.

The following sources were used:
	- keras helper functions
	- http://ai.stanford.edu/~amaas/data/sentiment/ Stanford Maas, ACL 2011 paper
		--> for the labeled data set
	- kaggle tutorial --> some text processing helper code