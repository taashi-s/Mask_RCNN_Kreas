"""
History Checkpoint Class Module
"""

from enum import Enum
import keras.callback as KC
from matplotlib import pyplot

class TargetHistory(Enum):
    """
    Target History
    """
    Loss = 0
    Accuracy = 1
    ValidationLoss = 2
    ValidationAccuracy = 3


class HistoryCheckpoint(KC.Callback):
    """
    History Checkpoint Class
    """

    def __init__(self, filepath, verbose=0, period=1, targets=None, is_each=True):
        super(HistoryCheckpoint, self).__init__()
        self.__verbose = verbose
        self.__filepath = filepath
        self.__period = period
        self.__epochs_since_last_save = 0
        self.__history_callback = KC.History()
        self.__targets = [TargetHistory.Loss]
        if isinstance(targets, list):
            self.__targets = targets
        self.__is_each = is_each


    def on_epoch_end(self, epoch, logs=None):
        self.__history_callback.on_epoch_end(epoch, logs)

        history = self.__history_callback.history
        logs = logs or {}
        self.__epochs_since_last_save += 1
        if self.__epochs_since_last_save >= self.period:
            self.__epochs_since_last_save = 0

            filepath = self.__filepath.format(epoch=epoch + 1)
            x = range(epoch)
            for target in self.__targets:
                key = self.get_history_key(target)
                pyplot.plot(x, history.history[key], label=key)
                pyplot.title(key)
                pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                if self.__is_each:
                    filepath = filepath.format(history=key)
                    pyplot.savefig(filepath)
                    pyplot.close()
            if not self.__is_each:
                pyplot.savefig(filepath)
                pyplot.close()

            if self.__verbose > 0:
                print('\nEpoch %05d: saving model .' % (epoch + 1))


    def get_history_key(self, target_history):
        key = ''
        if target_history == TargetHistory.Loss:
            key = 'loss'
        elif target_history == TargetHistory.Accuracy:
            key = 'acc'
        elif target_history == TargetHistory.ValidationLoss:
            key = 'val_loss'
        elif target_history == TargetHistory.ValidationAccuracy:
            key = 'val_acc'

        return key
