import torch
import numpy as np
from model import ConvRNN

def model_inference(test_loader, weights_path, model_params):
    model = ConvRNN(**model_params)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x
            batch_y = batch_y
            output = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().numpy())
            true.append(batch_y.detach().numpy())
    preds = np.concatenate(preds)
    true = np.concatenate(true)

    return preds, true