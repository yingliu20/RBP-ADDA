import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
from utils import *
from cnn_utils import *
from tf_util import *
from tensorflow.contrib import rnn
from tqdm import tqdm
initializer = tf.contrib.layers.xavier_initializer()
#initializer = tf.truncated_normal_initializer(stddev=0.1)

scale = 3.0
offset = 0.0
class Discriminator(object):

    def __init__(self, is_training=True, name="discriminator"):
        self.is_training = is_training
        self.name = name
        self.reuse = False

    def __call__(self, hidden_layer):
        with tf.variable_scope(self.name,reuse=self.reuse)  :
            C = hidden_layer.get_shape()[-1]

            hidden_output = hidden_layer
            hidden_output = fc_layer(hidden_output, 1, scope="fc1", is_training=self.is_training, activation_fn=None,use_xavier=False,stddev=0.02)
            preds = tf.squeeze(hidden_output, axis=1)  #

        self.reuse = True
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.name)

        for k in self.variable:
            print(k)

        return preds

class Predictor(object):

    def __init__(self, is_training=True, name="predictor"):
        self.is_training = is_training
        self.name = name
        self.reuse = False

    def __call__(self, hidden_layer):
        use_bn = False
        with tf.variable_scope(self.name,reuse=self.reuse)  :
            C = hidden_layer.get_shape()[-1]
            WFC1 = tf.get_variable(name='WFC1', shape=[C, C//2], initializer=initializer)
            BFC1 = tf.get_variable(name='BFC1', shape=[C//2], initializer=tf.constant_initializer(0))

            WFC2 = tf.get_variable(name='WFC2', shape=[C//2, 1],initializer=initializer)
            BFC2 = tf.get_variable(name='BFC2', shape=[1], initializer=tf.constant_initializer(0))


            # First fully connected layer: from flat CNN output to hidden layer
            hidden_output = fc_layer(hidden_layer,C//2,scope="fc0",bn=use_bn,is_training=self.is_training)
            #hidden_output = dropout_layer(hidden_output, keep_prob=0.5, is_training=self.is_training, scope='dp0')
            hidden_output = fc_layer(hidden_output,1,scope="fc1",is_training=self.is_training,activation_fn=None)
            preds = tf.squeeze(hidden_output, axis=1)  #

        self.reuse = True
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.name)
        for k in self.variable:
            print(k)

        return preds

class CNN_Net(object):

    def __init__(self, is_training=True, name="source_cnn"):
        self.is_training = is_training
        self.name = name
        self.reuse = False

    def __call__(self, data):
        with tf.variable_scope(self.name,reuse=self.reuse):

            out_dim = 32
            filters = [4]
            pooled_outputs = []
            use_bn = False
            use_avg = False
            conv_num = 2
            for filter_num in filters:
                conv_feat = conv_layer(data, out_dim//2, [filter_num, 4], scope='conv0_%d'%filter_num, is_training=self.is_training, bn=use_bn)
                conv_feat = tf.nn.max_pool(conv_feat, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
                print("CONV OUTPUT", tf.shape(conv_feat), conv_feat.get_shape())
                for i in range(1,conv_num):
                    conv_feat = conv_layer(conv_feat, out_dim, [filter_num, 1], scope='conv%d_%d'%(i,filter_num), is_training=self.is_training, bn=use_bn)
                    conv_feat = tf.nn.max_pool(conv_feat, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
                    print("CONV OUTPUT", tf.shape(conv_feat), conv_feat.get_shape())

                filter_output = conv_feat

                print("Input Data", tf.shape(data), data.get_shape())
                print("CONV OUTPUT", tf.shape(conv_feat), conv_feat.get_shape())
                print("FILTER OUTPUT", tf.shape(filter_output), filter_output.get_shape())
                pooled_outputs.append(filter_output)


            cnn_output = tf.concat(pooled_outputs, 3)
            out_dim = int(out_dim * len(filters))
            if use_avg:
                out_dim = out_dim*2


            fc_input = tf.contrib.layers.flatten(cnn_output)
            print("FC INPUT", tf.shape(fc_input), fc_input.get_shape())

            #exit(0)
            C = fc_input.get_shape()[-1]

            out_dim = C
            hidden_output = fc_layer(fc_input,out_dim,scope="fc0",bn=use_bn,is_training=self.is_training)
            hidden_output = fc_layer(hidden_output,out_dim//2,scope="fc1",bn=use_bn,is_training=self.is_training)

        self.reuse = True
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        print("CNN-Net:")
        for k in self.variable:
            print(k)
        return hidden_output

class Model(object):
    def __init__(self,params):
        self.params = params

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        ## for train predictor
        self.rna_data = tf.placeholder(tf.float32, [None, self.params['max_seq_len'], self.params['nucleotide'],1])
        self.rna_lengths = tf.placeholder(tf.int32, [None])
        self.rna_labels = tf.placeholder(tf.float32, [None])

        ## for train adda
        self.source_rna_data = tf.placeholder(tf.float32, [None, self.params['max_seq_len'], self.params['nucleotide'], 1])
        self.source_rna_lengths = tf.placeholder(tf.int32, [None])
        self.source_rna_labels = tf.placeholder(tf.float32, [None])

        self.target_rna_data = tf.placeholder(tf.float32, [None, self.params['max_seq_len'], self.params['nucleotide'], 1])
        self.target_rna_lengths = tf.placeholder(tf.int32, [None])
        self.target_rna_labels = tf.placeholder(tf.float32, [None])


        # Set strides
        self.strides_x_y = (1, 1)


        self.train_step = None



    def loss_function(self, opt,source=True):
        if(opt == "step1"):

            name = "source" if source else "target"
            self.predictor = Predictor(is_training=self.is_training,name=name+"_predictor")
            self.CNN = CNN_Net(is_training=self.is_training,name=name+"_cnn")
            M = self.CNN(self.rna_data)
            preds = self.predictor(M)

            # Loss function
            loss_score = tf.nn.l2_loss(self.rna_labels - preds)
            loss = loss_score

            self.train_step = tf.train.AdamOptimizer(self.params['lr']).minimize(loss)

            # calc accuracy
            accuracy = tf.pow(pearson_correlation(preds, self.rna_labels), 1)
            return loss, accuracy


        if(opt == "step2"):
            self.source_CNN = CNN_Net(is_training=False,name="source_cnn")
            self.target_CNN = CNN_Net(is_training=self.is_training,name="target_cnn")

            source_M = self.source_CNN(self.source_rna_data)
            target_M = self.target_CNN(self.target_rna_data)
            self.discriminator = Discriminator(is_training=self.is_training)
            self.D_source_logits = self.discriminator(source_M)
            # print("--------------------------------------")
            self.D_target_logits = self.discriminator(target_M)
            ls_gan = True
            if ls_gan:
                self.d_loss_source = tf.reduce_mean(tf.square(self.D_source_logits - 1.0))
                self.d_loss_target = tf.reduce_mean(tf.square(self.D_target_logits))
            else:
                self.d_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_source_logits,
                                                                                     labels=tf.ones_like(
                                                                                         self.D_source_logits)))
                self.d_loss_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_target_logits,
                                                                                     labels=tf.zeros_like(
                                                                                         self.D_target_logits)))
            dis_cost = 0.5*(self.d_loss_source + self.d_loss_target)


            self.predictor = Predictor(is_training=False,name="source_predictor")

            preds = self.predictor(target_M)
            loss_score = tf.nn.l2_loss(self.target_rna_labels - preds)
            self.loss = loss_score/1.0
            if ls_gan:
                target_M_cost = tf.reduce_mean(tf.square(self.D_target_logits - 1.0))
            else:
                target_M_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_target_logits,
                                                                                       labels=tf.ones_like(
                                                                                           self.D_target_logits)))
            accuracy = tf.pow(pearson_correlation(preds, self.target_rna_labels), 1)

            return dis_cost, target_M_cost+self.loss, accuracy

        if(opt == "step3"):
            self.predictor = Predictor(is_training=self.is_training,name="source_predictor")
            self.source_CNN = CNN_Net(is_training=False,name="source_cnn")
            self.target_CNN = CNN_Net(is_training=False,name="target_cnn")
            source_M = self.source_CNN(self.source_rna_data)
            source_preds = self.predictor(source_M)
            source_loss_score = tf.nn.l2_loss(self.source_rna_labels - source_preds)
            source_accuracy = tf.pow(pearson_correlation(source_preds, self.source_rna_labels), 1)

            target_M = self.target_CNN(self.target_rna_data)
            target_preds = self.predictor(target_M) #
            target_loss_score = tf.nn.l2_loss(self.target_rna_labels - target_preds)
            target_accuracy = tf.pow(pearson_correlation(target_preds, self.target_rna_labels), 1)


            return source_loss_score, source_accuracy , target_loss_score, target_accuracy

    def source(self,source=True):

        name = "source" if source else "target"
        self.predictor = Predictor(is_training=self.is_training,name=name + "_predictor")
        self.CNN = CNN_Net(is_training=self.is_training,name=name + "_cnn")
        M = self.CNN(self.rna_data)
        preds = self.predictor(M)
        # Loss function
        loss_score = tf.nn.l2_loss(self.rna_labels - preds)
        loss = loss_score
        # calculate accuracy
        accuracy = tf.pow(pearson_correlation(preds, self.rna_labels), 1)
        return loss, accuracy

    def adda(self):
        self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

        self.predictor = Predictor(is_training=self.is_training,name="source_predictor")
        self.CNN = CNN_Net(is_training=self.is_training,name="target_cnn")
        target_M = self.CNN(self.target_rna_data)
        preds = self.predictor(target_M)
        # Loss function
        loss_score = tf.nn.l2_loss(self.target_rna_labels - preds)
        loss = loss_score
        # calculate accuracy
        accuracy = tf.pow(pearson_correlation(preds, self.target_rna_labels), 1)
        return loss, accuracy


    def adda_predictor(self,source=True):
        self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

        self.predictor = Predictor(is_training=self.is_training,name="source_predictor")
        name = "source" if source else "target"
        self.CNN = CNN_Net(is_training=self.is_training,name="%s_cnn"%name)
        target_M = self.CNN(self.rna_data)
        preds = self.predictor(target_M)
        # Loss function
        loss_score = tf.nn.l2_loss(self.rna_labels - preds)
        loss = loss_score
        # calculate accuracy
        accuracy = tf.pow(pearson_correlation(preds, self.rna_labels), 1)
        return loss, accuracy

class MBPredictor:
    '''
    Init all parameters of a convolution neural network (CNN) for protein-RNA binding prediction
    '''

    def __init__(self, data_paths,params, strides_x_y, network_name):  #

        # Store data paths
        self.source_seq_path = data_paths[0]
        self.target_seq_path = data_paths[1]

        # Store network parameters
        self.params = params


        # Init hyperparameters data
        self.opt_epoch = -1
        self.weights = None
        self.biases = None
        self.kpool_x_y = None

        # Set the same strides for all filters
        self.single_strides_x_y = strides_x_y
        self.strides_x_y = None

        # Set fully connecetd layer data
        self.fc_size = -1
        self.WFC1 = None
        self.BFC1 = None
        self.WFC2 = None
        self.BFC2 = None

        self.network_name = network_name

    '''
    Init all hyperparamers
    '''

    def init_hyperparams(self):
        # Compute initial weights, biases and k-pools according to filters data
        self.weights = list()
        self.biases = list()
        self.kpool_x_y = list()

    '''
    Evaluate performance of current model on a given dataset (validation/testing)
    '''

    def evaluate_performance(self, eval_data, eval_lengths, eval_labels, scope):

        # Create a new session
        tf.reset_default_graph()
        new_graph = tf.Graph()
        with tf.Session(graph=new_graph) as eval_sess:
            # Import the trained network
            network = tf.train.import_meta_graph(self.network_name + '.meta')

            # Load the trained parameters
            network.restore(eval_sess, self.network_name)

            # Construct feed data using placeholder names
            rna_data = new_graph.get_tensor_by_name(scope + "rnadata:0")
            rna_lengths = new_graph.get_tensor_by_name(scope + "rnalengths:0")
            rna_labels = new_graph.get_tensor_by_name(scope + "rnalabels:0")
            feed_dict = {rna_data: eval_data, rna_lengths: eval_lengths, rna_labels: eval_labels}

            # Access the evaluation metric
            eval_accuracy = new_graph.get_tensor_by_name(scope + "rnaaccuracy:0")
            result = eval_sess.run(eval_accuracy, feed_dict)

            return result

    '''
    Train a convolution neural network (CNN) for protein-RNA binding prediction
    '''



    def source_train(self,source=True):


        tf.reset_default_graph()
        data_path = self.source_seq_path if source else self.target_seq_path
        all_train_data, all_train_lengths, all_train_labels = read_sequence(data_path, self.params['max_seq_len'])
        all_train_data = np.expand_dims(all_train_data,axis=-1)
        all_train_labels = all_train_labels/ scale + offset

        data_path = data_path.replace("train", "test")
        validation_data, validation_lengths, validation_labels = read_sequence(data_path, self.params['max_seq_len'])
        validation_data = np.expand_dims(validation_data, axis=-1)
        validation_labels = validation_labels / scale + offset

        all_train_size = all_train_labels.shape[0]

        cg = Model(self.params)
        loss, accuracy = cg.loss_function(opt="step1",source=source)
        # separate parameters into source_cnn and predictor
        all_vars = tf.trainable_variables()
        name = "source" if source else "target"
        var_source = [k for k in all_vars if k.name.startswith("%s_cnn"%name)]
        var_cls = [k for k in all_vars if k.name.startswith("%s_predictor"%name)]


        with tf.Session() as sess:


            sess.run(tf.global_variables_initializer())
            saver_source = tf.train.Saver(var_source)
            saver_cls = tf.train.Saver(var_cls)

            for epoch_index in range(self.params['num_epochs']):
                batch_counter = 0

                per = np.random.permutation(all_train_labels.shape[0])
                new_all_train_data = all_train_data[per, :, :]
                new_all_train_labels = all_train_labels[per]
                new_all_train_lengths = all_train_lengths[per]
                dataset = Dataset(new_all_train_data, new_all_train_lengths, new_all_train_labels, self.params['batch_size'])

                while dataset.has_next_batch():
                    batch_counter += 1

                    # Read next batch
                    rna_batch, lengths_batch, labels_batch = dataset.next_batch()
                    feed_dict = {cg.rna_data: rna_batch,
                                 cg.rna_lengths: lengths_batch,
                                 cg.rna_labels: labels_batch,
                                 cg.is_training: True}

                    # Perform batch optimization
                    sess.run(cg.train_step, feed_dict=feed_dict)

                    # check if performance on trianing data is decreasing enough to stop training process
                    if batch_counter % self.params['stop_check_interval'] == 0:
                        # Compute batch loss and accuracy
                        train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

                        status = 'Train acc: {}\tTrain Loss: {}\t{}\t{}\t{}'\
                                                 .format(train_accuracy, train_loss,\
                                                         batch_counter, dataset.curr_ind, epoch_index)
                        print (status)


                dataset.reset()
                data_size = validation_data.shape[0]
                test_acc = 0.0
                best_test_acc = 0.0
                num_batches = data_size // self.params['batch_size']
                for i in tqdm(range(0, data_size, self.params['batch_size'])):
                    batch_counter = batch_counter + 1
                    feed_dict = {
                        cg.rna_data: validation_data[i:i + self.params['batch_size']],
                        cg.rna_lengths: validation_lengths[i:i + self.params['batch_size']],
                        cg.rna_labels: validation_labels[i:i + self.params['batch_size']],
                        cg.is_training: False
                    }

                    result = sess.run(accuracy, feed_dict)
                    test_acc = test_acc + result
                test_acc = test_acc / num_batches
                print("-------------test_accuracy:",test_acc)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    name = "source" if source else "target"
                    source_cnn_path = "models/%s_cnn/"%name+ self.network_name +"-%s.ckpt"%name
                    if source:
                        pre_target_cnn_path = "models/target_pretrain/"+ self.network_name +"-target.ckpt"
                        saver_source.save(sess, pre_target_cnn_path)

                    cls_path = "models/predictor/"+ self.network_name +"-%s-predictor.ckpt"%name
                    saver_source.save(sess, source_cnn_path)
                    saver_cls.save(sess, cls_path)


    '''
    Apply ADDA to update  target network
    '''
    def adda_train(self):
        print("adda_train: ")

        source_data, source_lengths, source_labels = read_sequence(self.source_seq_path, \
                                                                                 self.params['max_seq_len'])
        source_data = np.expand_dims(source_data,axis=-1)
        source_labels = source_labels/scale+offset

        target_data, target_lengths, target_labels = read_sequence(self.target_seq_path, \
                                                                                 self.params['max_seq_len'])
        target_data = np.expand_dims(target_data,axis=-1)
        target_labels = target_labels/scale + offset


        data_path = self.target_seq_path.replace("train","test")
        target_validation_data, target_validation_lengths, target_validation_labels = read_sequence(data_path, \
                                                                        self.params['max_seq_len'])
        target_validation_data = np.expand_dims(target_validation_data, axis=-1)
        target_validation_labels = target_validation_labels / scale + offset



        all_train_size = target_labels.shape[0]
        validation_size = int(all_train_size * self.params['validation_fold'])
        validation_size = 0
        # Construct the training set
        target_train_data = target_data[validation_size:, :, :, :]
        target_train_lengths = target_lengths[validation_size:]
        target_train_labels = target_labels[validation_size:]



        print("Run network: ")
        cg = Model(self.params)
        loss1, loss2, accuracy = cg.loss_function(opt="step2")



        all_vars = tf.trainable_variables()
        var_source = [k for k in all_vars if k.name.startswith("source_cnn")]
        var_target = [k for k in all_vars if k.name.startswith("target_cnn")]
        var_dis = [k for k in all_vars if k.name.startswith("discriminator")]

        var_cls = [k for k in all_vars if k.name.startswith("source_predictor")]

        # two optimizer for discriminator and target
        print("optimizer initial: ")

        optimizer1 = tf.train.AdamOptimizer(learning_rate=self.params['lr_D']).minimize(loss1, var_list=var_dis)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.params['lr_M']).minimize(loss2, var_list=var_target)


        # TODO: (1) regularization (2) dropout (3) Decay in learning rate

        with tf.Session() as sess:
            print("Session_start: ")

            source_cnn_path = "models/source_cnn/"+ self.network_name +"-source.ckpt"
            pre_target_cnn_path = "models/target_pretrain/"+ self.network_name +"-target.ckpt"
            cls_path = "models/predictor/"+ self.network_name +"-source-predictor.ckpt"

            sess.run(tf.global_variables_initializer())
            print("sess_run_initial")
            saver_source = tf.train.Saver(var_source)
            print("1: ")
            saver_source.restore(sess, source_cnn_path)  # read from exist files
            print("2: ")
            saver_cls = tf.train.Saver(var_cls)
            print("3: ")
            saver_cls.restore(sess, cls_path)
            print("4: ")
            saver_target = tf.train.Saver(var_target)
            print("5: ")
            saver_target.restore(sess, pre_target_cnn_path)
            print("6: ")
            saver_dis = tf.train.Saver(var_dis)
            print("Restore ")

            # saver_all = tf.train.Saver(all_vars)


            final_acc = 0.0
            for epoch_index in range(self.params["num_epochs"]):
                total_cost1 = 0.0
                total_cost2 = 0.0
                idx = np.arange(target_train_data.shape[0])
                np.random.shuffle(idx)
                target_train_data = target_train_data[idx,...]
                target_train_labels = target_train_labels[idx]
                target_train_lengths = target_train_lengths[idx]

                idx2 = np.arange(source_data.shape[0])
                np.random.shuffle(idx2)
                source_data = source_data[idx2]
                source_labels = source_labels[idx2]
                source_lengths = source_lengths[idx2]

                if(source_data.shape[0] > target_train_data.shape[0]):
                    data_size = target_train_data.shape[0]
                else:
                    data_size = source_data.shape[0]
                for i in range(0, data_size, self.params['batch_size']):
                    _, cost1 = sess.run([optimizer1, loss1], feed_dict={
                        cg.source_rna_data: source_data[i:i + self.params['batch_size']],
                        cg.target_rna_data: target_train_data[i:i + self.params['batch_size']],
                        cg.target_rna_labels: target_train_labels[i:i + self.params['batch_size']]
                    })

                    #print(i)

                    d_loss_source, d_loss_target,d_loss,t_loss,D_source_logits,D_target_logits,cls_loss= \
                        sess.run([cg.d_loss_source, cg.d_loss_target,loss1,loss2,cg.D_source_logits,cg.D_target_logits,cg.loss],
                                 feed_dict={
                        cg.source_rna_data: source_data[i:i + self.params['batch_size']],
                        cg.target_rna_data: target_train_data[i:i + self.params['batch_size']],
                        cg.target_rna_labels: target_train_labels[i:i + self.params['batch_size']],
                        cg.is_training: True
                    })


                    _, cost2 = sess.run([optimizer2, loss2], feed_dict={
                        cg.target_rna_data: target_train_data[i:i + self.params['batch_size']],
                        cg.target_rna_labels: target_train_labels[i:i + self.params['batch_size']],
                        cg.is_training: True
                    })

                    total_cost1 = total_cost1 + cost1
                    total_cost2 = total_cost2 + cost2

                print(d_loss_source, "\t", d_loss_target, "\t", d_loss, "\t", t_loss, "\t", cls_loss)


                print(epoch_index, "\tD_source_losits:\t", D_source_logits[0], "\tD_target_logits:\t",
                      D_target_logits[0])


                total_acc = 0.0
                batch_num = 0
                for i in tqdm(range(0, target_validation_data.shape[0], self.params['batch_size'])):
                    acc = sess.run(accuracy, feed_dict={
                        cg.target_rna_data: target_validation_data[i:i + self.params['batch_size']],
                        #cg.target_rna_lengths: self.params['batch_size'],
                        cg.target_rna_labels: target_validation_labels[i:i + self.params['batch_size']],
                        cg.is_training: False
                    })
                    total_acc = total_acc + acc
                    batch_num = batch_num +1

                print("----------------------------------------------------------Accuracy: ","\t\t\t\t",total_acc / batch_num)
                if total_acc / batch_num > final_acc:
                    final_acc = total_acc / batch_num

                    # adda_source_path = "models/adda_source/" + self.network_name + "-source.ckpt"
                    target_cnn_path = "models/adda_target/" + self.network_name + "-target.ckpt"
                    dis_path = "models/discriminator/" + self.network_name + "-discriminator.ckpt"
                    # all_path = "models/all/" + self.network_name + "-adda.ckpt"
                    saver_target.save(sess, target_cnn_path)
                    saver_dis.save(sess, dis_path)
                    # saver_all.save(sess, all_path)
                    # saver_source.save(sess, adda_source_path)

                threshold = 0.96
                if (total_acc / target_validation_data.shape[0] > threshold):
                    break


    '''
    fine-tune the task predictor from both source network and target network
    '''
    def adda_train_predictor(self):
        print("adda_train_predictor: ")

        source_data, source_lengths, source_labels = read_sequence(self.source_seq_path, \
                                                                                 self.params['max_seq_len'])
        source_data = np.expand_dims(source_data,axis=-1)
        source_labels = source_labels/scale + offset

        data_path = self.source_seq_path.replace("train", "test")
        source_validation_data, source_validation_lengths, source_validation_labels = read_sequence(data_path, \
                                                                                                         self.params[
                                                                                                             'max_seq_len'])
        source_validation_data = np.expand_dims(source_validation_data, axis=-1)
        source_validation_labels = source_validation_labels / scale + offset

        target_data, target_lengths, target_labels = read_sequence(self.target_seq_path, \
                                                                                 self.params['max_seq_len'])
        target_data = np.expand_dims(target_data,axis=-1)
        target_labels = target_labels/scale + offset

        data_path = self.target_seq_path.replace("train", "test")
        target_validation_data, target_validation_lengths, target_validation_labels = read_sequence(data_path, \
                                                                                                         self.params[
                                                                                                             'max_seq_len'])
        target_validation_data = np.expand_dims(target_validation_data, axis=-1)
        target_validation_labels = target_validation_labels / scale + offset

        print("Run network: ")
        cg = Model(self.params)
        loss1, accuracy1, loss2, accuracy2 = cg.loss_function(opt="step3")


        all_vars = tf.trainable_variables()
        var_source = [k for k in all_vars if k.name.startswith("source_cnn")]
        var_target = [k for k in all_vars if k.name.startswith("target_cnn")]
        var_cls = [k for k in all_vars if k.name.startswith("source_predictor")]

        optimizer1 = tf.train.AdamOptimizer(self.params['lr_C']).minimize(loss1+loss2, var_list=var_cls)
        with tf.Session() as sess:
            print("Session_start: ")
            source_cnn_path = "models/source_cnn/"+ self.network_name +"-source.ckpt"
            target_cnn_path = "models/adda_target/"+ self.network_name +"-target.ckpt"
            cls_path = "models/predictor/"+ self.network_name +"-source-predictor.ckpt"

            sess.run(tf.global_variables_initializer())
            print("sess_run_initial")
            saver_source = tf.train.Saver(var_source)
            print("1: ")
            saver_source.restore(sess, source_cnn_path)
            print("2: ")
            saver_cls = tf.train.Saver(var_cls)
            print("3: ")
            print("cls_path: ", cls_path)
            saver_cls.restore(sess, cls_path)
            print("4: ")
            saver_target = tf.train.Saver(var_target)
            print("5: ")
            saver_target.restore(sess, target_cnn_path)
            print("Restore ")

            for epoch_index in range(self.params["num_epochs2"]):
                total_cost1 = 0.0
                total_cost2 = 0.0
                total_cost = 0.0
                batch_counter = 0.0
                total_acc1 = 0.0
                total_acc2 = 0.0
                total_acc = 0.0
                # print("d_loss_source,d_loss_target,d_loss_source1,d_loss_target1: ")
                idx = np.arange(target_data.shape[0])
                np.random.shuffle(idx)
                target_data = target_data[idx]
                target_labels = target_labels[idx]
                target_lengths = target_lengths[idx]

                idx2 = np.arange(source_data.shape[0])
                np.random.shuffle(idx2)
                source_data = source_data[idx2]
                source_labels = source_labels[idx2]
                source_lengths = source_lengths[idx2]

                if(source_data.shape[0] > target_data.shape[0]):
                    data_size = target_data.shape[0]
                else:
                    data_size = source_data.shape[0]
                for i in range(0, data_size, self.params['batch_size']):
                    feed_dict1 = {
                        cg.source_rna_data: source_data[i:i + self.params['batch_size']],
                        cg.source_rna_labels: source_labels[i:i + self.params['batch_size']],
                        cg.target_rna_data: target_data[i:i + self.params['batch_size']],
                        cg.target_rna_labels: target_labels[i:i + self.params['batch_size']],
                        cg.is_training: True
                    }
                    sess.run(optimizer1, feed_dict1)

                batch_counter = 0
                if (source_validation_data.shape[0] > source_validation_data.shape[0]):
                    data_size = source_validation_data.shape[0]
                else:
                    data_size = source_validation_data.shape[0]
                final_acc = 0.0
                for i in range(0,data_size, self.params['batch_size']):
                    batch_counter = batch_counter + 1
                    feed_dict2 = {
                        cg.source_rna_data: source_validation_data[i:i + self.params['batch_size']],
                        cg.source_rna_labels: source_validation_labels[i:i + self.params['batch_size']],
                        cg.target_rna_data: target_validation_data[i:i + self.params['batch_size']],
                        cg.target_rna_labels: target_validation_labels[i:i + self.params['batch_size']],
                        cg.is_training: False
                    }

                    # Compute batch loss and accuracy
                    train_loss1, train_accuracy1 = sess.run([loss1, accuracy1], feed_dict=feed_dict2)
                    train_loss2, train_accuracy2 = sess.run([loss2, accuracy2], feed_dict=feed_dict2)
                    # statistics
                    total_cost1 = total_cost1 + train_loss1
                    total_cost2 = total_cost2 + train_loss2
                    total_cost = total_cost + (1.0 / 2) * (train_loss1 + train_loss2)
                    total_acc1 = total_acc1 + train_accuracy1
                    total_acc2 = total_acc2 + train_accuracy2
                    total_acc = total_acc + (1.0 / 2) * (train_accuracy1 + train_accuracy2)

                # print("final_train_accuracy: ", train_accuracy)
                print("Cost: ", epoch_index,"\t", total_cost1 / batch_counter,"\t", total_cost2 / batch_counter,"\t",
                    "Accuracy: ", total_acc1 / batch_counter, "\t",total_acc2 / batch_counter)
                print("epoach: ",epoch_index,"total_acc: ", total_acc/batch_counter)

                if total_acc1 / batch_counter > final_acc:
                    final_acc = total_acc1 / batch_counter
                    cls_path = "models/adda_predictor/" + self.network_name + "-predictor.ckpt"
                    saver_cls.save(sess, cls_path)


                threshold = 0.96
                if (total_acc / target_data.shape[0] > threshold):
                    break



    '''
    Test pre-trained network
    '''

    def source_test(self,source=True):

        data_path = self.source_seq_path if source else self.target_seq_path

        test_data, test_lengths, test_labels = read_sequence(data_path,self.params['max_seq_len'])
        test_data = np.expand_dims(test_data, axis=-1)


        cg = Model( self.params)
        _, accuracy = cg.source(source=source)

        # parameter for source_cnn and predictor
        name = "source" if source else "target"
        all_vars = tf.global_variables()
        var_source = [k for k in all_vars if k.name.startswith("%s_cnn"%name)]
        var_cls = [k for k in all_vars if k.name.startswith("%s_predictor"%name)]

        with tf.Session() as eval_sess:

            print("1:")
            saver_source = tf.train.Saver(var_source)
            print("2:")
            saver_cls = tf.train.Saver(var_cls)
            print("3:")
            source_cnn_path = "models/%s_cnn/"%name + self.network_name + "-%s.ckpt"%name
            cls_path = "models/predictor/" + self.network_name + "-%s-predictor.ckpt"%name
            print(source_cnn_path)
            print(cls_path)
            saver_source.restore(eval_sess, source_cnn_path)
            print("4:")
            saver_cls.restore(eval_sess, cls_path)
            print("5:")
            data_size = test_data.shape[0]
            total_acc=0.0
            batch_counter = 0
            test_once = True
            if test_once:
                feed_dict = {cg.rna_data: test_data,
                             cg.rna_lengths: test_lengths,
                             cg.rna_labels: test_labels,
                             cg.is_training: False}
                total_acc = eval_sess.run(accuracy, feed_dict)
                batch_counter = 1.0
            else:
                for i in tqdm(range(0, data_size, self.params['batch_size'])):
                    batch_counter = batch_counter + 1
                    feed_dict = {
                        cg.rna_data: test_data[i:i + self.params['batch_size']],
                        cg.rna_lengths: test_lengths[i:i + self.params['batch_size']],
                        cg.rna_labels: test_labels[i:i + self.params['batch_size']],
                        cg.is_training: False
                    }

                    result = eval_sess.run(accuracy, feed_dict)
                    total_acc = total_acc + result
            print("7:")
            print("acc:",total_acc/batch_counter)
            return total_acc/batch_counter

    '''
    Test ADDA network
    '''

    def adda_test(self,data_path):

        test_data, test_lengths, test_labels = read_sequence(data_path, \
                                                                  self.params['max_seq_len'])
        test_data = np.expand_dims(test_data, axis=-1)
        test_labels = test_labels/scale + offset
        cg = Model(self.params)
        _, accuracy = cg.adda()

        # parameter for cnn and predictor
        all_vars = tf.global_variables()
        var_target = [k for k in all_vars if k.name.startswith("target_cnn")]
        var_cls = [k for k in all_vars if k.name.startswith("source_predictor")]

        with tf.Session() as eval_sess:

            print("1:#")
            saver_target = tf.train.Saver(var_target)
            print("2:#")
            saver_cls = tf.train.Saver(var_cls)
            print("3:#")
            target_cnn_path = "models/adda_target/" + self.network_name + "-target.ckpt"
            cls_path = "models/predictor/" + self.network_name + "-source-predictor.ckpt"
            print("target_cnn_path: ",target_cnn_path)
            print("cls_path: ", cls_path)
            print("4:#")
            saver_cls.restore(eval_sess, cls_path)
            print("5:#")
            saver_target.restore(eval_sess, target_cnn_path)
            print("6:#")

            feed_dict = {cg.target_rna_data: test_data, cg.target_rna_lengths: test_lengths, cg.target_rna_labels: test_labels}
            result = eval_sess.run(accuracy, feed_dict)
            print("8:#")

            return result

    '''
    Test predictor
    '''

    def adda_predictor_test(self,source=True):

        data_path = self.source_seq_path if source else self.target_seq_path
        test_data, test_lengths, test_labels = read_sequence(data_path, \
                                                                  self.params['max_seq_len'])
        test_data = np.expand_dims(test_data, axis=-1)
        test_labels = test_labels/scale + offset
        cg = Model(self.params)
        _, accuracy = cg.adda_predictor(source=source)
        # parameter for cnn and predictor
        name = "source" if source else "target"
        all_vars = tf.global_variables()
        var_target = [k for k in all_vars if k.name.startswith("%s_cnn"%name)]
        var_cls = [k for k in all_vars if k.name.startswith("source_predictor")]

        with tf.Session() as eval_sess:
            print("1:#")
            saver_target = tf.train.Saver(var_target)
            print("2:#")
            saver_cls = tf.train.Saver(var_cls)
            print("3:#")
            target_cnn_path = "models/adda_%s/"%name + self.network_name + "-%s.ckpt"%name
            cls_path = "models/adda_predictor/" + self.network_name + "-predictor.ckpt"
            print("target_cnn_path: ",target_cnn_path)
            print("cls_path: ", cls_path)
            print("4:#")
            saver_cls.restore(eval_sess, cls_path)
            print("5:#")
            saver_target.restore(eval_sess, target_cnn_path)
            print("6:#")

            feed_dict = {cg.rna_data: test_data,
                         cg.rna_lengths: test_lengths,
                         cg.rna_labels: test_labels}
            print("7:#")
            result = eval_sess.run(accuracy, feed_dict)
            print("result:",result)
            print("8:#")

            return result
