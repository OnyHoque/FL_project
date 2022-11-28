import matplotlib.pyplot as plt
import argparse
from DataProcessing import getNodes, makeNodeMalicious
from Model import getModel
import random
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--epoch', type=str, required=True)
parser.add_argument('--mnode', type=str, required=True)
parser.add_argument('--position', type=str, required=True)



args = parser.parse_args()

nodes, x_test, y_test = getNodes()
model = getModel(args.model)

epoch = int(args.epoch)
mnode = int(args.mnode)
position = args.position

if position == "front":
    for node_obj in nodes[:mnode]:
        makeNodeMalicious(node_obj)

elif position == "end":
    starting = len(nodes) - mnode
    for node_obj in nodes[starting:]:
        makeNodeMalicious(node_obj)

elif position == "random":
    for node_obj in nodes[:mnode]:
        makeNodeMalicious(node_obj)
    random.shuffle(nodes)

elif position == "none":
    print("Running federated learning without malicious nodes.")

else:
    print("Peleze, give correct position argument!")
    exit()
    

total_loss = []

for node_obj in nodes:
    node_obj.send_model(model)
    total_loss = total_loss + node_obj.train(epoch)
    model = node_obj.get_model()

plt.plot(total_loss)
plt.savefig('loss_graph.png')

joblib.dump(model, args.model+".model")