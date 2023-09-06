def segmentation_trials(X, fs, segs= 6):
  """
  Input
  X: numpy array with EEG info, dims (trials,channels,time)
  fs: int value with the sampling frequency
  segs: int vale with the length of the trial

  out
  X_train: numpy array with the eeg data windowed in trials with length "segs"
  """

  X_train = []
  windows = int(X.shape[2]/(segs*fs))
  for i in range(X.shape[0]):
    for j in range(windows):
        #print(j*(6*fs),int((j+1)*(segs*fs)))
        X_train.append(X[i,:,j*(segs*fs):int((j+1)*(segs*fs))])
  return np.asarray(X_train)

def kappa(y_true, y_pred):
    return cohen_kappa_score(np.argmax(y_true, axis = 1),np.argmax(y_pred, axis = 1))