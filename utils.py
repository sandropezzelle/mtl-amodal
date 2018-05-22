import warnings

from keras.callbacks import ModelCheckpoint


def read_qprobs(path):
    count = 0
    ratios = {}
    ratio_l = []

    with open(path + 'Q-probabilities.txt', 'r') as in_file:
        for line in in_file:
            els = line.strip().split('\t')

            if count == 0:
                for el in els[1:]:
                    ratios[el] = {}

                ratio_l = els[1:]

                for el in ratios:
                    for i in range(9):
                        ratios[el][str(i)] = 0.0
            else:
                ind = els[0]

                for i in range(17):
                    val = els[2 + i]
                    r = ratio_l[i]
                    ratios[r][ind] = val

            count += 1

    return ratios


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)
        self.last_saved_filename = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.last_saved_filename = filepath
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
