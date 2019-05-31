import tensorflow as tf
import numpy as np

class SEQ2SEQ_ATT:
    def __init__( self,  n_hidden_size, n_class, lr, n_enc_size, n_dec_size, n_enc_vocab_size, n_dec_vocab_size, n_embedding_size ):
        with tf.variable_scope('Input'):
            self.lr = lr
            self.n_class = n_class
            self.n_enc_size = n_enc_size
            self.n_dec_size = n_dec_size
            self.n_hidden_size = n_hidden_size
            self.n_enc_vocab_size = n_enc_vocab_size
            self.n_dec_vocab_size = n_dec_vocab_size
            self.n_embedding_size = n_embedding_size

            with tf.variable_scope('Placeholder'):
                self.enc_input = tf.placeholder( tf.int64, [None, None], name='enc_input')
                self.dec_input = tf.placeholder( tf.int64, [None, None], name='dec_input')
                self.inf_input = tf.placeholder( tf.int64, [None, None], name='inf_input' ) 
                self.targets   = tf.placeholder( tf.int64, [None, None], name='tar_input')
                self.x_seq_len = tf.placeholder( tf.int64, [None],       name="x_seq_len")
                self.y_seq_len = tf.placeholder( tf.int64, [None],       name="y_seq_len")
                self.dropout_keep = tf.placeholder( tf.float32,          name="dropout_keep")

            with tf.variable_scope('Variable'):
                # enc_embeddings [ enc_voc_size, embedding_size ]
                # dec_embeddings [ dec_voc_size, embedding_size ]
                self.enc_embeddings = tf.Variable(tf.random_normal([ self.n_enc_vocab_size, self.n_embedding_size]), name='enc_embedding')
                self.dec_embeddings = tf.Variable(tf.random_normal([ self.n_dec_vocab_size, self.n_embedding_size]), name='dec_embedding')

            with tf.variable_scope('MakeCell'):
                self.enc_cell = tf.nn.rnn_cell.LSTMCell( num_units=self.n_hidden_size )
                self.dec_cell = tf.nn.rnn_cell.LSTMCell( num_units=self.n_hidden_size )
                self.enc_cell = tf.nn.rnn_cell.DropoutWrapper( self.enc_cell, output_keep_prob=self.dropout_keep )
                self.dec_cell = tf.nn.rnn_cell.DropoutWrapper( self.dec_cell, output_keep_prob=self.dropout_keep )

            with tf.variable_scope('Embedding'):
                #enc_embed [ batch, seqlen, embedding_size ]
                self.enc_embed = tf.nn.embedding_lookup( self.enc_embeddings, self.enc_input, name='enc_embed' ) # ( enc_voc_size, hidden )
                self.dec_embed = tf.nn.embedding_lookup( self.dec_embeddings, self.dec_input, name='dec_embed' ) # ( dec_voc_size, hidden )

       # enc_state  [ 2,     batch,  hidden ] context, hidden 
       # enc_outputs[ batch, seqlen, hidden ]
        with tf.variable_scope('Encoder'):
            self.enc_outputs, self.enc_state = \
            tf.nn.dynamic_rnn( self.enc_cell, self.enc_embed, sequence_length=self.x_seq_len, dtype=tf.float32 )
            self.dec_state = self.enc_state


        # dec_embed [ batch, seqlen, hidden ]
        # context   [ batch, hidden         ]
        with tf.variable_scope('Decoder'):
            self.context = self.bahdanau_attention( self.enc_state, self.enc_outputs )
            self.t_dec_embed = tf.transpose( self.dec_embed, [1, 0, 2] );
            dec_idx=tf.constant(0)
            dec_output_tensor = tf.TensorArray( tf.float32, size = self.n_dec_size )
            def dec_cond( idx, p_state, enc_outputs, outupt_tensor, max_dec_size ):
                return tf.less( idx, max_dec_size )
        
            def dec_body( idx, p_state, enc_outputs, dec_output_tensor, max_dec_size ):
                i_dec_embed = tf.gather_nd(self.t_dec_embed, [[idx]])   
                i_dec_embed = tf.transpose(i_dec_embed, [1, 0, 2] )   # [batch, 1, hidden]
                context_expand = tf.expand_dims( self.context , 1)    # [batch, 1, hidden]
                i_dec_embed_concat = tf.concat( [ context_expand, i_dec_embed], axis=-1 )  # [ batch, 1, hidden*2 ]
                i_dec_outputs, i_dec_state = tf.nn.dynamic_rnn( self.dec_cell, i_dec_embed_concat, initial_state=p_state, dtype=tf.float32)
                self.context = self.bahdanau_attention( i_dec_state, self.enc_outputs )
                i_dec_outputs = tf.reshape( i_dec_outputs, [-1, self.n_hidden_size])
                dec_output_tensor = dec_output_tensor.write( idx, i_dec_outputs )
                return idx+1, i_dec_state, enc_outputs, dec_output_tensor, max_dec_size
        

        self.n_dec_state = tf.nn.rnn_cell.LSTMStateTuple(c=self.context, h=self.dec_state.h)
        with tf.variable_scope('While'):
            _, _, _, dec_output_tensor, _ = \
            tf.while_loop( cond = dec_cond, 
                           body = dec_body, 
                           loop_vars=[ dec_idx,
                                       self.n_dec_state, 
                                       self.enc_outputs, 
                                       dec_output_tensor,
                                       self.n_dec_size ] )

            self.dec_outputs = dec_output_tensor.stack()
            self.dec_outputs = tf.transpose(self.dec_outputs, [1, 0, 2] )
            self.logits = tf.layers.dense( self.dec_outputs, self.n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_dense')

    
        self.mask = tf.sequence_mask( self.y_seq_len, n_dec_size );
        with tf.variable_scope('Loss'):
           # targets [ batch, dec_voc_size ]
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=self.logits, labels=self.targets ) # losses =  [1, 32, 13]
            self.t_loss =  tf.boolean_mask( self.losses, self.mask )  
            self.loss = tf.reduce_mean( tf.boolean_mask( self.losses, self.mask )  )
            self.optimizer  = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('Accuracy'):
            self.prediction = tf.argmax(self.logits, 2, name='prediction', output_type=tf.int64 )
            prediction_mask = self.prediction * tf.to_int64(self.mask)
            correct_pred = tf.equal(prediction_mask, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")

         
        with tf.variable_scope('While'):
            def inf_cond( inf_idx, dec_input_idx, prev_state, output_tensor, max_dec_size ) :
                return tf.less( inf_idx, max_dec_size )

            def inf_body( inf_idx, dec_input_idx, prev_state, output_tensor, max_dec_size ) :
                dec_input_embeddings = tf.nn.embedding_lookup( self.dec_embeddings, dec_input_idx )   # [ batch, 1, embedding ] [ 
                context_expand = tf.expand_dims(self.context, 1)                                      # [ batch, 1, hidden    ]
                dec_input_embeddings = tf.concat( [ context_expand, dec_input_embeddings ], axis=-1 )
                dec_outputs, dec_state = tf.nn.dynamic_rnn( self.dec_cell, dec_input_embeddings, sequence_length=[1], initial_state=prev_state, dtype=tf.float32)
                self.context = self.bahdanau_attention( dec_state, self.enc_outputs )
                logits = tf.layers.dense( dec_outputs, self.n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_dense')
                idx_prediction = tf.argmax( logits, 2, output_type=tf.int64, name='idx_prediction' )
                output_tensor = output_tensor.write( inf_idx, idx_prediction )
                return inf_idx+1, idx_prediction, dec_state, output_tensor, max_dec_size

            inf_idx=tf.constant(0)
            inf_output_tensor = tf.TensorArray( tf.int64, size = self.n_dec_size, name='inf_output_tensor' )
            self.context = self.bahdanau_attention( self.enc_state, self.enc_outputs )
            self.n_dec_state = tf.nn.rnn_cell.LSTMStateTuple(c=self.context, h=self.dec_state.h)

            _, _, _, inf_output_tensor, _ = \
            tf.while_loop( cond = inf_cond, 
                           body = inf_body, 
                           loop_vars=[ inf_idx,
                                       self.inf_input,
                                       self.n_dec_state,
                                       inf_output_tensor,
                                       self.n_dec_size ])
            self.inf_result = inf_output_tensor.stack()
            self.inf_result = tf.reshape( self.inf_result, [-1], 'inf_result' ) 


    def bahdanau_attention( self, enc_state, enc_outputs ):
        query = enc_state.h             # ( 2,     batch, hidden )
        value = enc_outputs             # ( batch, seq,   hidden )
        query_exp = tf.expand_dims(query, 1) # ( 2,     1,     hidden )
        self.value_weight = tf.layers.dense(value,     self.n_hidden_size, activation=None, reuse=tf.AUTO_REUSE, name='value_weight')
        self.query_weight = tf.layers.dense(query_exp, self.n_hidden_size, activation=None, reuse=tf.AUTO_REUSE, name='query_weight')
        activation = tf.nn.tanh(self.value_weight + self.query_weight)
        attention_score = tf.layers.dense(activation, 1, reuse=tf.AUTO_REUSE, name='attention_score')
        attention_weight= tf.nn.softmax( attention_score )
        self.context = tf.reduce_sum( attention_weight * value, axis=1, name='attention_context' ) # context = [batch, hidden]
        return self.context;
 
