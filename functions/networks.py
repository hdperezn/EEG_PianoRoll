from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

def EEGNet(nb_classes, Chans = 64, Samples = 128,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   name='Conv2D_1',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   name='Depth_wise_Conv2D_1',
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    block2       = SeparableConv2D(F2, (1, 16),
                                   name='Separable_Conv2D_1',
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten(name = 'flatten')(block2)

    dense        = Dense(nb_classes, name = 'output',
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'out_activation')(dense)

    return Model(inputs=input1, outputs=softmax)

from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
from sklearn.base import  BaseEstimator, TransformerMixin, ClassifierMixin

class autoencoder_Pianoroll(BaseEstimator, ClassifierMixin):
  def __init__(self, img_size, loss, num_classes = 1, labels = 2, epochs=50,batch_size=32,
               learning_rate=1e-3,validation_split=0.2,verbose=1, droprate = 0.5, filters_list = [128, 64, 32],
               l1_l2 = 0, plot_loss= True):

    self.img_size = img_size
    self.labels = labels
    self.loss = loss 
    self.num_classes = num_classes
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate=learning_rate
    self.validation_split = validation_split
    self.verbose = verbose
    self.droprate = droprate
    self.plot_loss = plot_loss
    self.filters_list = filters_list
    self.l1_l2 = l1_l2

  def Encoder(self, img_size ):
    self.encoder_inputs = tf.keras.Input(shape=(img_size))
    ### [First half of the network: downsampling inputs] ###
    filters = self.filters_list[::-1]

    # Entry block
    x = layers.Conv2D(filters[0], 3, strides=2, padding="same", name = 'conv_filter32_1')(self.encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for fil in filters[1::]:
        x = layers.Dropout(self.droprate)(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(fil, 3, padding="same", name= 'Enco_conv_filter'+str(fil)+'_2')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(fil, 3, padding="same", name= 'Enco_conv_filter'+str(fil)+'_3',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(fil, 1, strides=2, padding="same", name= 'Enco_conv_filter'+str(fil)+'_4'
        , kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(previous_block_activation )
        x = layers.add([x, residual], name = 'Enco_add_block_filter'+str(fil))  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(fil, 3, strides= 2, padding="same", name= 'Enco_conv_filter'+str(fil)+'_5',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
    self.x_nonflat = x
    x = layers.Flatten()(x)
    self.x = x
    self.encoder = tf.keras.Model(self.encoder_inputs, outputs = [self.x], name="encoder")
    return self.encoder

  def Decoder(self, num_classes):
    input_x = tf.keras.Input(shape=(int(self.x.shape[1])))

    x = layers.Reshape((8, 4, 128))(input_x)
    x = layers.Conv2DTranspose(128, 3, strides= 2, padding="same", name= 'Deco_conv_filter'+str(128)+'_0',
                            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
    previous_block_activation = x

    for filters in self.filters_list:
        x = layers.Dropout(self.droprate)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", name= 'Deco_conv_filter'+str(filters)+'_1')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", name= 'Deco_conv_filter'+str(filters)+'_2',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same", name= 'Deco_conv_filter'+str(filters)+'_3',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    self.outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)
    self.decoder = tf.keras.Model(inputs= [input_x], outputs = self.outputs, name="Decoder")
    return self.decoder

  def get_model(self, *_):
    seed = 123
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()

    self.winitializer = tf.keras.initializers.GlorotNormal(seed=seed)
    self.binitializer = "zeros"
    # ---- call layers -----
    enco = self.Encoder(self.img_size)
    #cka = self.cka_model
    deco = self.Decoder(self.num_classes)
    # ---- def red ---------
    block = enco(self.encoder_inputs)
    decoder_ = deco(block)
    # ----- MODEL -------
    metris = [tf.keras.metrics.Recall(), tf.keras.metrics.SpecificityAtSensitivity(0.1)]

    self.model = tf.keras.Model(inputs=[self.encoder_inputs],  outputs=[decoder_], name = 'AE_CKA_MIDI')
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    #binady cross entropy
    #loss_classification = tf.keras.losses.BinaryCrossentropy()
    self.model.compile(loss= [self.loss] ,  optimizer= opt, metrics=metris)
    return

  def fit(self, X, Y, *_):
    # Y must be a list with the MIDI and the label
    callback1 = tf.keras.callbacks.TerminateOnNaN()

    self.get_model()
    self.history = self.model.fit(X, Y , epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split,
                                  callbacks=[callback1],
                                  verbose=self.verbose)
    # ----- plot loss -----
    if self.plot_loss:
          self.plt_history()

  def predict(self, X,  *_):
    return self.model.predict(X)
  def plt_history(self):
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return