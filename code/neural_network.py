from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
# global var
folder_path = "/Users/alex/Desktop/UofT/CSC311/Final_Project"

def load_data(base_path=folder_path):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    print("train_matrix")
    print(train_matrix)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        #out = inputs

        out = self.g(inputs)
        out = F.sigmoid(out)
        out = self.h(out)
        out = F.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: all_epoch, all_loss, all_valid_acc
    """
    # Add a regularizer to the cost function.
    all_epoch = []
    all_valid_acc = []
    all_loss = []

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + \
                   (lamb/2)*(model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        all_epoch.append(epoch)
        all_loss.append(train_loss)
        all_valid_acc.append(valid_acc)

    return all_epoch, all_loss, all_valid_acc
#####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()


    # # finding best k using the validation set
    # k = [10, 50, 100, 200, 500]
    # lr = 0.01
    # num_epoch = 1
    # lamb = 0
    # for k_i in k:
    #     model = AutoEncoder(train_matrix.shape[1], k_i)
    #     # Set optimization hyperparameters.
    #
    #     print("K is {}".format(k_i))
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch)

    # Based on validation accuracy, k is 200 is selected

    # # finding best learning rate and number of epoch
    # lr = [0.0001, 0.001, 0.01]
    # num_epoch = [100]
    # lamb = 0
    #
    # k = 200
    #
    # for lr_i in lr:
    #     for num_epoch_i in num_epoch:
    #         model = AutoEncoder(train_matrix.shape[1], k)
    #         # Set optimization hyperparameters.
    #
    #         print("K is 200, learning rate is {}, number of epoch is {}".format(
    #                 lr_i, num_epoch_i))
    #         train(model, lr_i, lamb, train_matrix, zero_train_matrix,
    #               valid_data, num_epoch_i)

    # to optimize validation accuracy, lr 0.01 is selected, and num_epoch 60 is
    # selected

    # Plot the graphs
    k_best = 200
    lr_best = 0.01
    num_epoch_best = 60
    # lamb = 0
    # model = AutoEncoder(train_matrix.shape[1], k_best)
    # all_epoch, all_loss, all_valid_acc = train(model, lr_best, lamb, train_matrix,
    #                                            zero_train_matrix,
    #                                            test_data, num_epoch_best)
    #
    #
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    #
    # axes[0].plot(all_epoch, all_loss, label="Training Loss vs Epoch")
    # # axes[0].title("Training Loss vs Epoch")
    # axes[0].set_xlabel('Epoch')
    # axes[0].set_ylabel('Loss')
    #
    # axes[1].plot(all_epoch, all_valid_acc, label="Validation Accuracy vs. Epoch")
    # # axes[1].title("Validation Accuracy vs. Epoch")
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('Accuracy')
    #
    # plt.legend()
    # plt.show()
    # plt.savefig("q3d.png")

    # Q3e tuning parameter lambda
    # lamb = [0.001, 0.01, 0.1, 1]
    #
    # for lamb_i in lamb:
    #     model = AutoEncoder(train_matrix.shape[1], k_best)
    #     print("Lambda is {}".format(lamb_i))
    #     all_epoch, all_loss, all_valid_acc = train(model, lr_best, lamb_i,
    #                                                train_matrix,
    #                                                zero_train_matrix,
    #                                                valid_data, num_epoch_best)

    # best lambda is 0.01
    lamb_best = 0.01
    model = AutoEncoder(train_matrix.shape[1], k_best)
    train(model, lr_best, lamb_best, train_matrix, zero_train_matrix, valid_data, num_epoch_best)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
