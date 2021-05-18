# Deep-Learning-FNN-from-scratch
## Feedforward Neural Network and Backpropagation From Scartch(with no Keras、Tensorflow、Pytorch)

交大電信所 簡仁宗教授 HW1-1 

### 作業說明:

You are given the dataset of medical images (MedMNIST.zip). (Dataset url:https://reurl.cc/6yMKGk)
This dataset contains 6 classes. In this exercise, you need to implement a feedforward neural network (FNN) model by
yourself to recognize radiological images, and use specified algorithms to update the parameters. 
Please use train.npz as training data and test.npz as test data.

![image](https://user-images.githubusercontent.com/45477381/114274155-bbdf4e80-9a4f-11eb-962a-6bd69e353ab3.png)

### 執行結果1:

模型設計:
Hidden layer1: 64 nodes/ 
Hidden layer2: 32 nodes/ 
output layer: 6 nodes。 / 
Epoch number = 20， Batch size = 2048， Learning rate = 0.8。

Training Process:

![image](https://user-images.githubusercontent.com/45477381/114274394-ca7a3580-9a50-11eb-8ff6-c863a4526a9d.png)

Learning curves of Loss and the Accuracy:

![image](https://user-images.githubusercontent.com/45477381/114274376-b20a1b00-9a50-11eb-927b-136d9493f8b6.png)

### 執行結果2:

模型設計同上，將batch size改為 = 1024

Training Process:

![image](https://user-images.githubusercontent.com/45477381/114274513-4aa09b00-9a51-11eb-8fb6-086b9ff34d06.png)

Learning curves of Loss and the Accuracy:

![image](https://user-images.githubusercontent.com/45477381/114274518-512f1280-9a51-11eb-9b32-d0b2dbcf1284.png)
