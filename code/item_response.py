from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    log_lklihood = 0.
    df = pd.DataFrame.from_dict(data)
    for i, row in df.iterrows():
        question = row["question_id"]
        user = row["user_id"]
        log_lklihood += (row["is_correct"] * (theta[user] - beta[question])
                         - np.log1p(np.exp(theta[user] - beta[question])))
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """

    theta1 = np.zeros(542)
    theta1 = theta.copy()
    beta1 = np.zeros(1774)
    beta1 = beta.copy()
    all_uid = np.array(data["user_id"])
    all_qid = np.array(data["question_id"])

    for i in range(len(theta)):
        derv_theta = 0
        row_lst = np.where(all_uid == i)[0]
        for row in row_lst:
            j = data["question_id"][row]
            cij = data["is_correct"][row]
            sig_input = (theta[i] - beta[j]).sum()
            derv_theta += cij - sigmoid(sig_input)
        theta1[i] += lr*derv_theta

    for j in range(len(beta)):
        derv_beta = 0
        row_lst = np.where(all_qid == j)[0]
        for row in row_lst:
            i = data["user_id"][row]
            cij = data["is_correct"][row]
            sig_input = (theta[i] - beta[j]).sum()
            derv_beta += -cij + sigmoid(sig_input)
        beta1[j] += lr*derv_beta

    return theta1, beta1


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    np.random.seed(311)
    hyperparameter = np.random.random(1)[0]
    theta_rand = np.random.rand(542) * hyperparameter
    beta_rand = np.random.rand(1774) * hyperparameter

    train_acc_lst = []
    train_log_likelihood = []
    val_acc_lst = []
    val_log_likelihood = []

    i = 0
    while i <= iterations:
        train_neg_likelihood = neg_log_likelihood(data, theta=theta_rand, beta=beta_rand)
        train_log_likelihood.append(train_neg_likelihood)
        valid_neg_likelihood = neg_log_likelihood(val_data, theta=theta_rand, beta=beta_rand)
        val_log_likelihood.append(valid_neg_likelihood)

        train_eval = evaluate(data=data, theta=theta_rand, beta=beta_rand)
        train_acc_lst.append(train_eval)
        valid_eval = evaluate(data=val_data, theta=theta_rand, beta=beta_rand)
        val_acc_lst.append(valid_eval)

        print("NLLK: {} \t train acc: {} \t validate acc {}".format(train_neg_likelihood, train_eval, valid_eval))
        theta_rand, beta_rand = update_theta_beta(data, lr, theta_rand, beta_rand)
        i += 1

    return theta_rand, beta_rand, train_log_likelihood, val_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.02
    print("The current learning rate is: {}". format(lr))
    iteration = 50
    iter = []
    for i in range(51):
        iter.append(i)
    theta, beta, train_log_likelihood, \
        val_log_likelihood = irt(train_data, val_data, lr, iteration)

    plt.plot(iter, train_log_likelihood, label='training')
    plt.plot(iter, val_log_likelihood, label='Validation')
    plt.xlabel("iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Negative log likelihood as a function of iterations")
    plt.legend()
    plt.show()


    validate_acc = evaluate(val_data, theta=theta, beta=beta)
    test_acc = evaluate(test_data, theta=theta, beta=beta)
    print("The Accuracy for validation set is: {}".format(validate_acc))
    print("\nThe Accuracy for test set is {}".format(test_acc))

    questions = np.array([13, 23, 45])
    theta = theta.reshape(-1)
    theta.sort()
    for question in questions:
        diff = beta[question]
        prob = sigmoid(theta - diff)
        plt.plot(theta, prob, label="Question No.{}".format(question))
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.title("Probability of the correct response as a function of theta for three questions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
