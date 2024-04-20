import os

PORT = 5050
IP_ADDR = '127.0.0.1'
NUMBER_OF_CLIENTS = 4
INPUT_SHAPE = (32, 32, 3)
CYCLES = 4
EPOCHS = 50
HEADER_SEND = 10
HEADER_RECV = 11
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = '!DISCONNECT'

FLAGS = {
    'ID': '0',
    'CONNECTED': '1',
    'GLOBAL WEIGHTS': '2',
    'LOCAL WEIGHTS': '3',
    '!DISCONNECT': '4',
}
INV_FLAGS = {
    '0': 'ID',
    '1': 'CONNECTED',
    '2': 'GLOBAL WEIGHTS',
    '3': 'LOCAL WEIGHTS',
    '4': '!DISCONNECT',
}
PWD = os.getcwd()
INIT_WEIGHTS = os.path.join(PWD, 'init_weights.pkl')

DATA = os.path.join(PWD, 'data')

CLIENT_PATH = os.path.join(DATA, 'FEDERATED')
TEST_SERVER_PATH = os.path.join(CLIENT_PATH, 'test')

def log_cleansing():
    for file in os.listdir():
        if file.endswith('log') or file.endswith('pkl') or file.endswith('h5'):
            os.remove(file)
