import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

def calc_rrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    mean_y_true = np.mean(y_true)

    rrmse = rmse / mean_y_true
    return rrmse

class WSB:
    def __init__(self, strong_predictor=None, weak_predictors=None, weight_g=1.0):
        self.weight_g = weight_g
        self.strong_predictor = strong_predictor
        self.weak_predictors = weak_predictors


    def fit(self, x, y):
        for predictor in [self.strong_predictor] + self.weak_predictors:
            predictor.fit(x, y)

    def predict(self, x, weight_t=1.0):
        preds = []

        for i in range(x.shape[0]):
            strong_pred = float(self.strong_predictor.predict(x)[0])
            weak_preds = np.array([float(wp.predict(x)[0]) for wp in self.weak_predictors])

            mean_preds = np.mean(weak_preds)
            std_preds = np.std(weak_preds)
            z_score = (weak_preds - mean_preds) / std_preds
            weak_preds = weak_preds[np.abs(z_score) < 1.0]

            if len(weak_preds) == 0:
                preds.append(strong_pred)
                continue
            else:
                mean_preds = np.mean(weak_preds)

            booster = (strong_pred - mean_preds) * self.weight_g * weight_t
            final_pred = strong_pred + booster
            preds.append(final_pred)

        return preds
