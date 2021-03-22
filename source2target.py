import sys, getopt

import tensorflow as tf
import os
from glob import glob


usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    print(checkpoint_dir)
    checkpoint = checkpoint_dir#tf.train.get_checkpoint_state(checkpoint_dir)
    print("che: ",checkpoint)
    #exit(0)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            print("Var_name: ")
            print(var_name)
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
                print("###########1")
            if add_prefix:
                new_name = add_prefix + new_name
                print("#############2")

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
                print("###########3")
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                print("#########4")
                # Rename the variable
                var = tf.Variable(var, name=new_name)
            print("new_name: ")
            print(new_name)
            print("replace_from",replace_from," replace_to",replace_to)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint)


def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)
    print("Start: ")
    modellist = glob(os.path.join(checkpoint_dir,"*.ckpt.meta"))
    print(modellist)
    for model in modellist:
        model_name = model[:-5]
        print(model_name)
        rename(model_name, replace_from, replace_to, add_prefix, dry_run)
    print("end: ",replace_from,replace_to)
    print("End: ")

if __name__ == '__main__':
    main(sys.argv[1:])