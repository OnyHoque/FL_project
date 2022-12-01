import matplotlib.pyplot as plt
import argparse
from DataProcessing import getNodes, makeNodeMalicious
from Model import getModel
from matplotlib.pyplot import figure
import random
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    for i in range(len(nodes)):
        nodes[i].node_number = i

elif position == "none":
    print("Running federated learning without malicious nodes.")

else:
    print("Peleze, give correct position argument!")
    exit()
    

total_loss = []

weights = model.get_weights()

for node_obj in nodes:
    model.set_weights(weights)
    node_obj.send_model(model)
    total_loss, weights = total_loss + node_obj.train(epoch)
    model = node_obj.get_model()

figure(figsize=(10, 6), dpi=80)
plt.plot(total_loss)
image_name = "loss_graph"
image_name += "_" + args.model + "_" + args.mnode + "_" + args.position + ".png"
plt.savefig(image_name)

joblib.dump(model, args.model+".model")

result = model.evaluate(x_test, y_test)
print("Accuracy:", result[1]*100, "%")

# y_pred = model.predict(x_test)

# y1 = y_test
# y2 = y_pred.argmax(1)
# print('Precision: %.3f' % precision_score(y1, y2, average='micro'))
# print('Recall: %.3f' % recall_score(y1, y2, average='micro'))
# print('F1: %.3f' % f1_score(y1, y2, average='micro'))
# print('Accuracy: %.3f' % accuracy_score(y1, y2))