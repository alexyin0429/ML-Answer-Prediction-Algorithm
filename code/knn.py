import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy for student similarity: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    nbrs = KNNImputer(n_neighbors=k)
    mat = np.transpose(nbrs.fit_transform(np.transpose(matrix)))
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy for item similarity: {}".format(acc))
    return acc


def plot_accuracy(k_list, acc, title):
    plt.plot(k_list, acc)
    plt.xlabel("k")
    plt.xticks(k_list)
    plt.grid(axis='x', color='0.95')
    plt.title(title)
    plt.show()


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    val_acc_user = []
    val_acc_question = []
    for k in k_values:
        print("k is {}".format(k))
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_question = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc_user.append(acc_user)
        val_acc_question.append(acc_question)

    plt.plot(k_values, val_acc_user)
    plt.legend(["Accuracy for user-based collaborative filtering"])
    plt.xlabel("The value of k")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(k_values, val_acc_question)
    plt.legend(["Accuracy for item-based collaborative filtering"])
    plt.xlabel("The value of k")
    plt.ylabel("Accuracy")
    plt.show()

    # For user-based filtering, the k resulting the highest acc is 11
    k_star = 11
    print("\nThe k for best performance of user-based collaborative filtering is {}".format(k_star))
    final_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("The final test accuracy for user-based collaborative filtering is {}".format(final_acc))

    # For item-based filtering, the k resulting the highest acc is 21
    k_star_item = 21
    print("\nThe k for best performance of item-based collaborative filtering is {}".format(k_star_item))
    final_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)
    print("The final test accuracy for item-based collaborative filtering is {}".format(final_acc_item))



if __name__ == "__main__":
    main()
