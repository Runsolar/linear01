from numpy import array
import random


def train_test_split(x_input, y_input, test_percent, mixing):
    x_train, x_test, y_train, y_test = [], [], [], []

    train_percent = 1 - test_percent

    if mixing:

        mixed = list(zip(x_input, y_input))
        random.shuffle(mixed)

        mixed_train = mixed[:int(train_percent*len(mixed))]
        mixed_test = mixed[int(train_percent*len(mixed)):]

        for val1, val2 in zip(mixed_train, mixed_test):

            x_train.append(val1[0])
            y_train.append(val1[1])
            x_test.append(val2[0])
            y_test.append(val2[1])

    else:

        x_train, x_test = x_input[:int(train_percent*len(x_input))], x_input[int(train_percent*len(x_input)):]
        y_train, y_test = y_input[:int(train_percent*len(y_input))], y_input[int(train_percent*len(y_input)):]

    return array(x_train), array(y_train), array(x_test), array(y_test)