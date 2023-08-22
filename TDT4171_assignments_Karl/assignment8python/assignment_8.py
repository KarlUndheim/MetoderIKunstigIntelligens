import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
from tensorflow import keras
from keras.utils import pad_sequences



def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//8
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy

    epochs = 1
    model = keras.Sequential()
    
    if model_type=="feedforward":
        # 2 epochs is enough because of the small size of the network.
        epochs = 2
        model.add(
            keras.layers.Embedding(
                input_dim=data["vocab_size"],
                # Output_dim was set to 4 arbitrarily, this seems to work well.
                output_dim=4,
                # I increased the size of input_length to increase accuracy.
                input_length=data["max_length"]//8
            )
        )
        # Flatten to fit into dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(
            128,
            activation="relu",
            # One of many possible initializers, most of which are probably equally good for this dataset.
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros"
        ))
        # Between layers I tried adding batch normalization, but neither the accuracy or convergence time increased so I removed them.
        # Dropout marginally increased the accuracy.
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(
            512,
            activation="relu",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros"
        ))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros",
        ))
        model.add(keras.layers.Dropout(0.1))

        # The last layer consists only of one unit which determines if the input article is positive or negative.
        model.add(keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros"
        ))
        
    else:
        model.add(
            keras.layers.Embedding(
                input_dim=data["vocab_size"],
                output_dim=4,
                input_length=data["max_length"]//8
            )
        )
        # The recurrent network
        model.add(keras.layers.LSTM(3, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(3, return_sequences=True))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.BatchNormalization())
        # Flatten for the final layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros"
        ))

    model.compile(
        # Learning rate down to 0.001 from 0.005 increased accuracy a little bit
        # Adam performed much better than SGD
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss = keras.losses.binary_crossentropy,
        metrics = keras.metrics.binary_accuracy
    )
    train = model.fit(
        x = data["x_train"],
        y = data["y_train"],
        epochs=epochs,
    )
    loss, accuracy = model.evaluate(data["x_test"], data["y_test"], batch_size=64)
    model.summary()

    return accuracy



def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()

