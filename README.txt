#To run the code, you need to open terminal from the local directory.


Command:
python -m main --model=Generic --epoch=10 --mnode=10 --position=front

--model:    Using this parameter you can pass the name of the model. All valid options are:
            ResNet50
            DenseNet121
            MobileNet
            VGG16
            InceptionV3
            EfficientNetB7
            Generic

--epoch:    It is the epoch used for training by each nodes

--mnode:    It represents the percentage of malicious nodes. A valid input is in the range of 0 to 100.

--position: Using this parameter you can pass the position of the malicous nodes in the training process. All valid options are
            front   : To place all malicious nodes in the front
            end     : To place all malicious nodes in the end
            random  : To randomly place malicious nodes in various parts of the training phase
            none    : To train the FL model cleanly without any malicous nodes



# How to run the code:
You need to save the graphs by running the following code and changing the --mnode from 10 to 100 by a increment of 10 steps.
python -m main --model=Generic --epoch=10 --position=front --mnode=0


