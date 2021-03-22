# RBP-ADDA
RBP-ADDA: Inferring RNA-binding protein target preferences based on adversarial domain adaptation

	
Requirements:

	Python 3.7, Numpy 1.19, Tensorflow 1.15.

Setting up:

	Clone the repositopry into your working space.
	
RUN the model:
1. Source_train: python main.py source_train test_data/train
pre-train a source network and task predictor; initialize target network by sharing same parameters and architecture with source network

2. Source_test: python main.py source_test test_data/test
use pre-train network to test source data

3. ADDA_train: python main.py adda_train test_data/train
update the parameters of target network based on adversarial domain adaptation

4. ADDA_test: python main.py adda_test test_data/test
use the updated target network to test target data

5. ADDA_predictor: python main.py adda_predictor test_data/train
fine-tune the parameters of task predictor with both source network and target network.

6. ADDA_predictor_test_source: python main.py adda_predictor_test_source test_data/test
use the source network and new task predicor to test source data 

7. ADDA_predictor_test_target: python main.py adda_predictor_test_target test_data/test
use the updated target network and new task predicor to test target data 


File names:

	Source data file should have the string "source" in its name.
	Target data file should have the string "target" in its name.
	Training and testing files of the same experiment should have the same name (in two different directories).

Input format:

A file contains a set of sequnces together with their estimated binding affinities. Every line in the files starts with binding affinity and then followed by the RNA sequence. For example:

	0.027569	AGAAGGCACCAACAGAAGCUCUAACCAGACUAGCCACC


Every experiment has four possible related files (two for training and two for testing).
