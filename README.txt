Train LSTM on one of two tasks:
- Run the LSTM_keras script for binary classification
- Run the LSTM_categories script for numerical rating prediction

TO RUN:
As stated above, run 'LSTM_keras.py' to process or load the necessary
data files as well as build, train, and test the binary classification
LSTM model on the processed imdb dataset. Run 'LSTM_categories.py' to do
so for the rating classification task. The resulting model weights,
intermediate activations produced by the test and training sets,
class predictions, and prediction accuracy will be pickled and sored
in a folder labeled 'results'.  

Run 'run_classifiers.py' to compare
how well two implementations (including sklearn's implementation) of
the k-Nearest Neighbors classifier perform on the intermediate
activations of the binary classification task.

Running the binary classification task will yield slightly
different accuracy percentages each time, even though the data
set can be reproducibly randomized with a seed. The randomized
selection of batches in the actual training of the model leads
to the accuracy variation across trials. The accuracy of the
trials was generally above 80%, though in some trials it was
as low as 66.5%.

The following sources were used:
	- keras helper functions
	- keras LSTM imdb example
	- http://ai.stanford.edu/~amaas/data/sentiment/ Stanford Maas, ACL 2011 paper
		--> for the labeled data set
	- kaggle tutorial
		--> some text processing helper code