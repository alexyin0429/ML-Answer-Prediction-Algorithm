from knn import *
from neural_network import *
from item_response import *


def bagging(data):
    # Randomize an index
    bootstrap_data = {"user_id": [],
                      "question_id": [],
                      "is_correct": []}
    for i in range(len(data["question_id"])):
        rand_index = np.random.randint(0, len(data["question_id"]))
        bootstrap_data["question_id"].append(data["question_id"][rand_index])
        bootstrap_data["user_id"].append(data["user_id"][rand_index])
        bootstrap_data["is_correct"].append(data["is_correct"][rand_index])
    print("bootstrap data")
    print(bootstrap_data)
    return bootstrap_data


def change_dic_to_sparse(bootstrap_data, num_q, num_s):
    # change bootstrap_data to sparse matrix
    bootstrap_matrix = np.empty((num_s, num_q), )
    bootstrap_matrix[:] = np.nan
    print('original')
    print(bootstrap_matrix)
    for i in range(len(bootstrap_data["question_id"])):
        q_id = bootstrap_data["question_id"][i]
        u_id = bootstrap_data["user_id"][i]
        # bootstrap_matrix[u_id][q_id] = bootstrap_data["is_correct"][i]
        pos = [int(num_q * u_id + q_id)]
        np.put(bootstrap_matrix, pos, bootstrap_data["is_correct"][i])
        # mat = mat.transpose().tocsr()
    return bootstrap_matrix


"""
Returns a list of 0 or 1 predictions for each test data observation, using
bootstrap matrix and neural network to train.
"""


def neural_net_predictions(bootstrap_matrix, test_data, k, num_epoch, lamb, lr):

    # bootstrap_matrix = bootstrap_matrix.toarray()
    # print("neural_network_matrix")
    # print(bootstrap_matrix)
    zero_bootstrap_matrix = bootstrap_matrix.copy()
    # Fill in the missing entries to 0.
    zero_bootstrap_matrix[np.isnan(bootstrap_matrix)] = 0
    # print("zero bootstrap matrix")
    # print(zero_bootstrap_matrix)
    # Change to Float Tensor for PyTorch.
    zero_bootstrap_matrix = torch.FloatTensor(zero_bootstrap_matrix)
    bootstrap_matrix = torch.FloatTensor(bootstrap_matrix)

    #k = 200
    #lr = 0.01
    #num_epoch = 1
    #lamb = 0.01
    model = AutoEncoder(bootstrap_matrix.shape[1], k)
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = bootstrap_matrix.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.
        # print("epoch " + str(epoch))
        for user_id in range(num_student):
            inputs = Variable(zero_bootstrap_matrix[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)
            # if epoch == num_epoch - 1:
            # print("last output")
            # print(output)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(bootstrap_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + \
                   (lamb/2)*(model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

    # y_pred = model(inputs)
    # print("prediction")
    # print(y_pred)
    # Tell PyTorch you are evaluating the model.
    model.eval()

    pred_lst = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_bootstrap_matrix[u]).unsqueeze(0)
        output = model(inputs)
        # print("output")
        # print(output)
        guess = output[0][test_data["question_id"][i]].item() >= 0.5
        # print("guess decimal: " + str(output[0][test_data["question_id"][i]].item()))
        # print("guess: " + str(guess))
        pred_lst.append(guess)

    return pred_lst


def knn_impute_by_user_val(matrix, valid_data, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy for student similarity: {}".format(acc))
    return acc


def knn_impute_by_user_pred(spars_matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(spars_matrix)
    return mat
    # test_data has keys ['user_id', 'question_id', 'is_correct']
    # user_id is row number, qs_id is coloumn number in spars_matrix


def knn_prediction(mat, test_data) -> list:
    prediction = []
    for i in range(len(test_data['user_id'])):
        row = test_data['user_id'][i]
        column = test_data['question_id'][i]
        prediction.append(mat[row][column])
    return prediction


def train_irt(data, lr, iteration):
    np.random.seed(311)
    hyperparameter = np.random.random(1)[0]
    theta_rand = np.random.rand(542) * hyperparameter
    beta_rand = np.random.rand(1774) * hyperparameter
    i = 0
    while i <= iteration:
        print("iteration {}".format(i))
        theta_rand, beta_rand = update_theta_beta(data, lr, theta_rand, beta_rand)
        i += 1
    return theta_rand, beta_rand


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # For neural network
    print("Neural Network Begins")
    num_q = sparse_matrix.shape[1]
    num_s = sparse_matrix.shape[0]
    bootstrap_data = bagging(train_data)
    bootstrap_matrix = change_dic_to_sparse(bootstrap_data, num_q, num_s)
    k = 10
    lr = 0.01
    lamb = 0.001
    num_epoch = 100
    pred_by_neural_val = neural_net_predictions(bootstrap_matrix,
                                                val_data, k, num_epoch,
                                                lamb, lr)
    print('Neural Net validation parameters for k: {}, lr: {}, '
          'lambda: {}, num_epoch: {}'.format(k, lr, lamb, num_epoch))

    correct = 0
    for i in range(len(val_data["question_id"])):
        if pred_by_neural_val[i] == val_data["is_correct"][i]:
            correct += 1
    print("This is the validation acc : {}".format(correct / len(val_data["question_id"])))
    pred_by_nwrk = neural_net_predictions(bootstrap_matrix, test_data)
    print("Neural Network ends")

    # For item-response model
    print("IRT begins")
    lr = 0.02
    print("The current learning rate is: {}". format(lr))
    iteration = 15
    # ## train IRT model
    theta, beta = train_irt(bootstrap_data, lr, iteration)
    validate_acc = evaluate(val_data, theta=theta, beta=beta)
    print("This is the validation acc : {}".format(validate_acc))
    # Make prediction on test dataset
    pred_by_irt = []
    data = test_data
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        prediction = sigmoid(x)
        pred_by_irt.append(int(prediction >= 0.5))
    print("IRT ends")

    # For kNN method
    print('KNN begins')
    # # For kNN method
    # # This will return the optimal k
    # k = knn_find_k(bootstrap_matrix, val_data)
    k_values = [1, 3, 6, 8, 11, 13, 16, 20, 21, 24, 26, 28, 31]
    val_acc_user = []
    for k in k_values:
        acc_user = knn_impute_by_user_val(sparse_matrix.toarray(), val_data, k)
        val_acc_user.append(acc_user)
    big_index = val_acc_user.index(max(val_acc_user))
    k_star = k_values[big_index]
    print("The optimal K we choose is {}".format(k_star))
    acc = max(val_acc_user) # Get the validation accuracy
    print("Validation Accuracy for student similarity: {}".format(acc))
    mat1 = knn_impute_by_user_pred(bootstrap_matrix, k_star)
    pred_by_knn = knn_prediction(mat1, test_data)


    # calculate the overall prediction
    overall_pred = []
    for i in range(len(data["question_id"])):
        pred_1 = pred_by_irt[i]
        pred_2 = pred_by_nwrk[i]
        pred_3 = 1 if pred_by_knn[i] >= 0.5 else 0

        if (pred_1 + pred_2 + pred_3) / 3 >= 0.5:
            overall_pred.append(1)
        else:
            overall_pred.append(0)

    correct = 0
    for i in range(len(data["question_id"])):
        if overall_pred[i] == test_data["is_correct"][i]:
            correct += 1
    acc = correct / len(data["question_id"])
    print("The accuracy on test dataset is {}".format(acc))
    return acc


if __name__ == '__main__':
    main()
