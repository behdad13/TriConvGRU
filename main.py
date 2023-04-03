import argparse
from preprocessing import prepare_datasets
from DataLoader import create_dataloader
from trainer import train_and_validate
from inference import model_inference
from preprocessing import prepare_test_datasets
from utils import calculate_metrics
from utils import create_forecast_table
from utils import plot_forecast


def main(
    train_csv = "data.csv",
    n_timesteps = 72,
    for_hor = 3,
    batch_size = 64,
    epoch = 10,
    patience = 20,
    num_feature = 1,
    test_csv = "test.csv",
):

    X_train, y_train, X_val, y_val = prepare_datasets(train_csv, train_size_ratio=0.8, valid_size_ratio=0.2, n_timesteps=n_timesteps, for_hor=for_hor)
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size)

    hyper = {'LR': [0.001, 0.0001], 'hidden1':[16, 32], 'hidden2':[16, 32]}
 
    best_hidden1, best_hidden2, best_lr, min_val_loss_total = train_and_validate(train_loader, val_loader, hyper, epochs=epoch, patience=patience, for_hor=for_hor, num_feat=num_feature, timestamp=n_timesteps)
    

    #for testing the model
    X_test, y_test = prepare_test_datasets(test_csv, n_timesteps=n_timesteps, for_hor=for_hor)
    test_loader = create_dataloader(X_test, y_test, batch_size=1, shuffle=False)


    model_params = {
    'input_dim': num_feature,
    'timesteps': n_timesteps,
    'output_dim': for_hor,
    'n_channels1': best_hidden1,
    'n_channels2': best_hidden1,
    'n_channels3': best_hidden1,
    'n_units1': best_hidden2,
    'n_units2': best_hidden2,
    'n_units3': best_hidden2}

    # getting the result
    preds, true = model_inference(test_loader, "model/TriConvGRU.pt", model_params)

    #create the result table
    df = create_forecast_table(preds, true)
    print(df.head())

    # calculate metrics and plot the result
    plot_forecast(df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_timesteps", type=int, default=72)
    parser.add_argument("--for_hor", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_feature", type=int, default=1)
    parser.add_argument("--patience",type=int,default=30)
    parser.add_argument("--epoch",type=int,default=100)
    args = parser.parse_args()

    main(
        n_timesteps=args.n_timesteps,
        for_hor = args.for_hor,
        batch_size=args.batch_size,
        num_feature=args.num_feature,
        patience=args.patience,
        epoch=args.epoch
    )