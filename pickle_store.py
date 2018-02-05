import pickle


def store(imput_sth, filename):
    fw = open(filename, 'wb')
    pickle.dump(imput_sth, fw)
    fw.close()


def grab(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
