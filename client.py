from __init__ import HEADER_SEND, HEADER_RECV, INV_FLAGS, \
    PORT, FORMAT, IP_ADDR, DISCONNECT_MESSAGE, FLAGS, CLIENT_PATH, NUMBER_OF_CLIENTS, INPUT_SHAPE, EPOCHS, CYCLES
from utils import find_client_index, create_model, load_local_data, baseline_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import socket
import logging
import pickle
import os
import tensorflow as tf

class Client:
    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.model = baseline_model(INPUT_SHAPE, 'resnet50')
        self.compile_model()

    def connecting(self):
        logging.info('[CONNECTING] Connecting...')
        self.client_socket.connect((self.addr, self.port))

    def compile_model(self):
        self.model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])


    def send_msg(self, _flag, _message):
        def message_formatting(_message):
            pickled_message = pickle.dumps(_message)
            msg_bytes = bytes(''.join([FLAGS[_flag], f'{len(pickled_message):<{HEADER_SEND}}']),
                              FORMAT) + pickled_message
            return msg_bytes

        _message = message_formatting(_message)
        logging.info(f'[SENDING {_flag}] SENDING {_flag} TO SERVER')
        self.client_socket.send(_message)

    def receive_msg(self):
        l = []
        received_message = self.client_socket.recv(HEADER_RECV)
        HEADER = received_message.decode(FORMAT)
        _flag = HEADER[0]
        message_length = int(HEADER[1:])
        l.append(received_message)
        len_l = len(received_message)
        while True:
            received_message = self.client_socket.recv(HEADER_RECV)
            l.append(received_message)
            len_l += len(received_message)
            if len_l - HEADER_RECV == message_length:
                break
        full_message = b''.join(l)
        full_message = pickle.loads(full_message[HEADER_RECV:])
        logging.info(f'[RECEIVING MESSAGE]')
        return full_message, INV_FLAGS[_flag]


def main(index):
    client = Client(IP_ADDR, PORT)
    client.connecting()
    client.receive_msg()
    client.send_msg('CONNECTED', 'CONNECTED')
    connected = True
    batch_size = 64
    dataset = load_local_data(index)
    train = dataset['train'].batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val = dataset['val'].batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    checkpoint_callback_last = ModelCheckpoint(
        filepath=os.path.join('Results', f'{EPOCHS}EPOCHS_{CYCLES}CYCLES_{batch_size}BATCH', f'federated_model_client{index}_last.h5'),
        save_weights_only=False,
        save_best_only=False,
        verbose=1)
    checkpoint_callback_best = ModelCheckpoint(
        filepath=os.path.join('Results', f'{EPOCHS}EPOCHS_{CYCLES}CYCLES_{batch_size}BATCH', f'federated_model_client{index}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    accuracy, val_accuracy = [], []
    data_tosave = {}
    while connected:
        message, flag = client.receive_msg()
        if flag == 'GLOBAL WEIGHTS':
            logging.info('[GOT GLOBAL WEIGHTS]')
            client.model.set_weights(message)
            history = client.model.fit(train,
                        validation_data=val,
                        epochs=EPOCHS,
                        callbacks=[checkpoint_callback_last, checkpoint_callback_best])
            accuracy.extend(history.history['accuracy'])
            val_accuracy.extend(history.history['val_accuracy'])
            logging.info(history.history)
            client.send_msg('LOCAL WEIGHTS', client.model.get_weights())
        if flag == DISCONNECT_MESSAGE:
            data_tosave['accuracy'] = accuracy
            data_tosave['val_accuracy'] = val_accuracy
            pickle.dump(data_tosave, open(os.path.join('Results', f'{EPOCHS}EPOCHS_{CYCLES}CYCLES_{batch_size}BATCH', f'federated_model_client{index}_history.pkl'), 'wb'))
            break


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    INDEX = find_client_index() - 1
    logging.basicConfig(level=logging.INFO, filename=f'client{INDEX}.log', filemode='w',
                    format='%(message)s')

    main(INDEX)
