from adda_model import *
import time
import glob
import sys
import os
import tensorflow as tf

# Random generator initializers
tf.set_random_seed(1)
np.random.seed(1)
print('len(sys.argv): ',sys.argv)

if len(sys.argv) != 3:
    print ("Usage: python main.py source_train data/train")
    exit()


# Set model parameters
params = dict()
params['max_seq_len'] = 41
params['nucleotide'] = 4
params['batch_size'] = 256
params['beta'] = 0.001
params['lr'] = 0.001
params['lr_C'] = 0.00005
params['lr_D'] = 0.00005
params['lr_M'] = 0.001
params['num_epochs'] = 1000
params['num_epochs2'] = 1000
params['stop_check_interval'] = 50
params['early_stopping_memory'] = 3
params['validation_fold'] = 1.0 / 3.0
params['k-fold'] = 3

# Set the layers data
# layer1 = {'input_channels': 1, 'output_channels': 256}

# Set strides
strides_x_y = (1, 1)


# Set data paths
directive = sys.argv[1]
data_dir = sys.argv[2]
print('sys.argv[1]: ',sys.argv[1])
print('sys.argv[2]: ',sys.argv[2])


# Add trailing dir separator
if not data_dir.endswith(os.path.sep):
    data_dir = data_dir + os.path.sep
 

source_sequence_files = sorted(glob.glob(data_dir + '*source*'))
target_sequence_files = sorted(glob.glob(data_dir + '*target*'))

if len(source_sequence_files) == 0 or len(target_sequence_files) == 0:
    print ("Warning: no input files found!")
    exit()


for source_file, target_file in zip(source_sequence_files, target_sequence_files):
    print (source_file + "\n" + target_file)
    start_time = time.time()
    data_paths = (source_file, target_file)
    net_file = source_file[len(data_dir):-11]
    print("net_file:",net_file)
    predictor = MBPredictor(data_paths, params,  strides_x_y, net_file)

    if directive == "source_train":
        # Train model
        predictor.source_train(source=True)
        del predictor
        tf.reset_default_graph()
        print("change the source_cnn to target_cnn")
        os.system("python source2target.py --checkpoint_dir models/target_pretrain --replace_from source_cnn --replace_to target_cnn")


    elif directive == "source_test":
        # Test the model
        result = predictor.source_test(source=True)
        del predictor
        tf.reset_default_graph()


    elif directive == "adda_train":
        # Test the model
        predictor.adda_train()
        del predictor
        tf.reset_default_graph()

    elif directive == "adda_test":
        # Test the model
        result = predictor.adda_test(data_paths[1])
        print (result)
        del predictor
        tf.reset_default_graph()

    elif directive == "adda_predictor":
        predictor.adda_train_predictor()
        del predictor
        tf.reset_default_graph()
    elif directive == "adda_predictor_test_source":
        predictor.adda_predictor_test(source=True)
        del predictor
        tf.reset_default_graph()

    elif directive == "adda_predictor_test_target":
        predictor.adda_predictor_test(source=False)
        del predictor
        tf.reset_default_graph()

    else:
         print ("Unknown directive")
         break

    end_time = time.time()
    duration = end_time - start_time
    print (duration)
