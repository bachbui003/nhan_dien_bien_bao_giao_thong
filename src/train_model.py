from model import build_model
from data_preprocessing import load_data, split_data
from keras.models import load_model

def train_and_save_model():
    data, labels = load_data(classes=43)
    X_train, X_test, y_train, y_test = split_data(data, labels)
    
    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))
    
    model.save("my_model.h5")
    return history

if __name__ == '__main__':
    train_and_save_model()
