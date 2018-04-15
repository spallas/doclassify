import os

import tensorflow as tf
import numpy as np
import tqdm
import pickle
from tensorboard.plugins import projector
from data_preprocessing import generate_batch, build_dataset, save_vectors, read_analogies
from data_preprocessing import build_pairs, detect_phrases, read_and_preprocess
from evaluation_w2v import Evaluation

TMP_DIR = "./tmp"
DOMAIN_WORDS = 3_500_000
EVAL_RANGE = 150_000
DATASET_DUMP = "dataset35M.pkl"
TOKENS_LIST_DUMP = "raw_tokens35M.pkl"


class Word2Vec:

    def __init__(self, embedding_size, window_size,
                 neg_samples, vocabulary_size):
        """
        Initialize this model specific hyper-parameters. Training hyper-parameters
        will be given as arguments to the train method.
        :param embedding_size:
        :param window_size:
        :param neg_samples: number of negative samples
        :param vocabulary_size:
        """
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.vocabulary_size = vocabulary_size

        # Evaluation parameters:
        # the portion of the training set used for data evaluation
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)

        # fields to be populated by load_data method:
        self.data_ind = []
        # dataset expressed as a large list of integers representing
        # the index of a word in the following dictionaries
        self.word2int = {}
        self.int2word = {}
        self.questions = None  # will be an np.array of int32

        # evaluation graph, built during training
        self.evaluation = None

        # Clean old logs and training data
        for f_name in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, f_name))
        return

    def load_data(self, directories, analogies_file, ignore_cache=False):
        """
        Load data from a list of directories, or from cached pickle binaries
        :param ignore_cache:
        :param analogies_file:
        :param directories: list of directories containing
        :return:
        """
        if os.path.exists(DATASET_DUMP) and not ignore_cache:
            with open(DATASET_DUMP, "rb") as f:
                self.data_ind, self.word2int, self.int2word, self.questions = pickle.load(f)
        else:
            if os.path.exists(TOKENS_LIST_DUMP) and not ignore_cache:
                with open(TOKENS_LIST_DUMP, "rb") as f:
                    word_phrases_list = pickle.load(f)
            else:
                raw_tokens_list = []
                raw_pairs_list = []
                for directory in directories:
                    tokens, pairs = read_and_preprocess(directory, domain_words=DOMAIN_WORDS)
                    raw_tokens_list += tokens
                    raw_pairs_list += pairs
                print("LOG: len of raw_tokens_list:", len(raw_tokens_list))
                print("LOG: len of raw_pairs_list:", len(raw_pairs_list))
                word_phrases_list = detect_phrases(raw_tokens_list, raw_pairs_list)
                del raw_tokens_list  # Hint to reduce memory.
                del raw_pairs_list
                print("LOG: Again...")
                word_phrases_list = detect_phrases(word_phrases_list, build_pairs(word_phrases_list))

                # CACHE resulting python object:
                with open(TOKENS_LIST_DUMP, "wb") as f:
                    pickle.dump(word_phrases_list, f)

            # print("LOG: Number of words or phrases in preprocessed corpus: ", len(set(word_phrases_list)))

            # Build the pseudo-one-hot encoding of corpus plus dictionaries
            print("LOG: building dataset...")
            self.data_ind, self.word2int, self.int2word = build_dataset(word_phrases_list,
                                                                        self.vocabulary_size)
            del word_phrases_list
            print("LOG: Done.")

            # read the question file for the Analogical Reasoning evaluation
            print("LOG: Reading analogies...")
            self.questions = read_analogies(analogies_file, self.word2int)
            print("LOG: Done.")

            # CACHE resulting python object to make subsequent experiments faster
            with open(DATASET_DUMP, "wb") as f:
                pickle.dump([self.data_ind, self.word2int, self.int2word, self.questions], f)

        # update size with actual one, in case it is less
        # than the chosen upper bound:
        self.vocabulary_size = len(self.word2int)
        print("LOG: size of self.data_ind", len(self.data_ind))
        print("LOG: Size of dictionary: ", len(self.word2int))

    # USE THIS TO TRAIN MODEL:

    def train(self, batch_size=32, num_train_steps=80_000_000, start_learning_rate=1.0):
        """
        Define the model graph and perform training and saving of metadata and embeddings results.
        :param batch_size:
        :param num_train_steps:
        :param start_learning_rate:
        """
        graph = tf.Graph()
        with graph.as_default():
            # Define input data tensors.
            with tf.name_scope('inputs'):
                inputs = tf.placeholder(tf.int32, shape=[batch_size])
                labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
                global_step = tf.Variable(0, name='global_step', trainable=False)

            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.get_variable("embeddings_matrix",
                                             shape=[self.vocabulary_size, self.embedding_size],
                                             initializer=tf.random_uniform_initializer(-1.0, 1.0))
                embedding_output = tf.nn.embedding_lookup(embeddings, inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('loss'):
                weights = tf.get_variable("weights",
                                          shape=[self.vocabulary_size, self.embedding_size],
                                          initializer=tf.truncated_normal_initializer(
                                                stddev=1.0 / (self.embedding_size ** 0.5)))

                biases = tf.get_variable("bias", initializer=tf.zeros([self.vocabulary_size]))

                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=weights,
                        biases=biases,
                        labels=labels,
                        inputs=embedding_output,
                        num_sampled=self.neg_samples,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                       800_000, 0.987, staircase=True)

            # Construct the SGD optimizer using a learning rate of 1.0 and exponential decay.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Compute the cosine similarity between mini batch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

            # evaluation graph
            self.evaluation = Evaluation(normalized_embeddings, self.word2int, self.questions)

        # ========= TRAINING ========= #

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(TMP_DIR, session.graph)
            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')
            saver_step = num_train_steps // 20
            total_loss = 0
            bar = tqdm.tqdm(range(num_train_steps))
            for step in bar:
                batch_inputs, batch_labels = generate_batch(batch_size, step,
                                                            self.window_size,
                                                            self.data_ind)

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op
                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict={inputs: batch_inputs, labels: batch_labels},
                    run_metadata=run_metadata)
                total_loss += loss_val

                self._evaluation_current(writer, summary, step, similarity, total_loss,
                                         run_metadata, session, num_train_steps)

                if step % saver_step == 0 and step is not 0:
                    saver.save(session, os.path.join(TMP_DIR, "w2v"), global_step=global_step)
                    save_vectors(normalized_embeddings.eval().tolist())

            final_embeddings = normalized_embeddings.eval()

            # ========= SAVE VECTORS ========= #

            save_vectors(final_embeddings.tolist())

            # Write corresponding labels for the embeddings.
            with open(os.path.join(TMP_DIR, 'metadata.tsv'), 'w') as f:
                for i in range(self.vocabulary_size):
                    f.write(self.int2word[i] + '\n')

            # Save the model for checkpoints
            saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = 'metadata.tsv'
            projector.visualize_embeddings(writer, config)

        writer.close()

    def _evaluation_current(self, writer, summary, step, similarity,
                            total_loss, run_metadata, session, num_train_steps):
        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == 0:
            return
        if step % (EVAL_RANGE*4) == 0:
            sim = similarity.eval()
            for i in range(self.valid_size):
                valid_word = self.int2word[self.valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = self.int2word[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        if step == (num_train_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)
        if step % EVAL_RANGE is 0:
            print("\nEVALUATING ACCURACY...")
            self.evaluation.eval(session)
            print("avg loss: " + str(total_loss / step))
