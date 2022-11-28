class Node:
    node_number = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    node_type = "I am a good node"
    model = None

    def __init__(self, x_train, y_train, node_number):
        self.x_train = x_train
        self.y_train = y_train
        self.node_number = node_number


    def get_data(self):
        return self.x_train, self.y_train

    def set_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        self.node_type = "I am a bad node"

    def whatAmI(self):
        print(self.node_type)

    def send_model(self, model):
        self.model = model
    
    def train(self, epoch):
        history = self.model.fit(self.x_train, self.y_train, epochs=epoch, batch_size=100, verbose=0, validation_split=0.1)
        print("Node: ",self.node_number, " says:","Traning complete.")
        return history.history['loss']

    def get_model(self):
        return self.model





