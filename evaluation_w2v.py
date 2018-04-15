import tensorflow as tf


class Evaluation:
    def __init__(self, nemb, dictionary, questions):
        self.dictionary = dictionary
        self.questions = questions
        self.nemb = nemb
        self.vocab_size = len(dictionary.keys())
        self.build_eval_graph()
        self.log = open("./tmp/log_eval.txt", "w")
        self.reversed = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        #  nemb = tf.nn.l2_normalize(self.emb_layer, 1)
        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.nn.embedding_lookup(self.nemb, analogy_a)  # a's embs
        b_emb = tf.nn.embedding_lookup(self.nemb, analogy_b)  # b's embs
        c_emb = tf.nn.embedding_lookup(self.nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, self.nemb, transpose_b=True)

        # For each question (row in dist), find the top 10 words.
        _, pred_idx = tf.nn.top_k(dist, 10)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(self.nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, self.nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, self.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

    def _predict(self, session, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def eval(self, session):
        """Evaluate analogy questions and reports accuracy.
        """
        self.log = open("./tmp/log_eval.txt", "w")
        # How many questions we get right at precision@1.
        correct = 0

        try:
            total = self.questions.shape[0]
        except AttributeError as e:
            raise AttributeError("Need to read analogy questions.")

        start = 0
        while start < total:
            limit = start + 2500
            sub = self.questions[start:limit, :]
            idx = self._predict(session, sub)
            start = limit
            for question in range(sub.shape[0]):
                for j in range(4):
                    if idx[question, j] == sub[question, 3]:
                        # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        for k in range(4):
                            print(self.reversed[sub[question, k]], file=self.log, end="")
                        print(file=self.log)

                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
        self.log.close()
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                                  correct * 100.0 / total))
        return correct * 100.0 / total
