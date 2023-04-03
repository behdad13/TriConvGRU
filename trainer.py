import torch
import torch.nn as nn
import numpy as np
from model import ConvRNN

def train_and_validate(train_loader, val_loader, hyper, epochs=400, patience=30, for_hor=3, num_feat=2, timestamp=72):
    loss_fn = nn.MSELoss()
    min_val_loss_total = np.inf
    best_hidden_layer1 = 0
    best_hidden_layer2 = 0
    best_lr = 0

    for LR in hyper['LR']:
        for hidden1 in hyper['hidden1']:
            for hidden2 in hyper['hidden2']:
                model = ConvRNN(num_feat, timestamp, for_hor,
                                    n_channels1=hidden1, n_channels2=hidden1, n_channels3=hidden1,
                                    n_units1=hidden2, n_units2=hidden2, n_units3=hidden2).cuda()
                min_val_loss = np.inf
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                counter = 0

                for i in range(epochs):
                    mse_train = 0
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        opt.zero_grad()
                        y_pred = model(batch_x)
                        y_pred = y_pred.squeeze(1)
                        l = loss_fn(y_pred, batch_y)
                        l.backward()
                        mse_train += l.item() * batch_x.shape[0]
                        opt.step()

                    with torch.no_grad():
                        mse_val = 0
                        preds = []
                        true = []
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.cuda()
                            batch_y = batch_y.cuda()
                            output = model(batch_x)
                            output = output.squeeze(1)
                            preds.append(output.detach().cpu().numpy())
                            true.append(batch_y.detach().cpu().numpy())
                            mse_val += loss_fn(output, batch_y).item() * batch_x.shape[0]

                    if min_val_loss > mse_val:
                        min_val_loss = mse_val
                        counter = 0
                    else:
                        counter += 1

                    if counter == patience:
                        break

                    if min_val_loss_total > mse_val:
                        min_val_loss_total = mse_val
                        best_hidden_layer2 = hidden2
                        best_hidden_layer1 = hidden1
                        best_lr = LR
                        torch.save(model.state_dict(), "model/TriConvGRU.pt")

                print(f"with hidden layer1:{hidden1} | hidden layer2:{hidden2} | LR:{LR} --> the error equals:{(min_val_loss)} ")
                print("===========================================================")
                print("next hyperparameter combination")

    print(f'tuning is done==>> the best hidden layer1:{best_hidden_layer1}|the best hidden layer2:{best_hidden_layer2}|the best LR:{best_lr} --> minimum val loss:{(min_val_loss_total)} ')
    return best_hidden_layer1, best_hidden_layer2, best_lr, min_val_loss_total