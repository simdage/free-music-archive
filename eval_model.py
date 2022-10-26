

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.series
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    r2_score, mean_absolute_percentage_error, mean_pinball_loss, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler



class EvaluateModel:
    """
    The class ScikitLearnModels include functionalities to train, test and observe test results. It can use
    different types of scikit learn model, as long as parameters send to a random search are expected by the API.
    """

    def __init__(self, X_train, y_train, X_test, y_test, parameters, model_type, experiment_name, regression,
                 multiclass=False,
                 api=None):
        """
        @param X_train: pandas DataFrame
        @param y_train: pandas DataFrame
        @param X_test: pandas DataFrame
        @param y_test: pandas DataFrame
        @param parameters: dictionary of parameters for random search
        @param model_type: any scikit learn model (RandomForestRegressor, GradientBoostingClassifier, etc.)
        """
        self.model_type = model_type
        self.parameters = parameters
        self.columns = X_train.columns
        self.results = None
        self.param_used = None
        self.model = None
        self.models = None
        self.pred_actual = None
        self.scaler = None
        self.alpha = None
        self.api = api
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.regression = regression
        self.multiclass = multiclass
        self.experiment_name = experiment_name

    def get_metrics_multiclass(self, model):

        train_auc = roc_auc_score(y_train_dumm, y_train_pred_prob, multi_class=arg_)
        test_auc = roc_auc_score(y_test_dumm, y_test_pred_prob, multi_class=arg_)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')

        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,

        }

    def get_metrics(self):
        """
        use model to create predictions on test set, then create metrics and store metrics in dict
        @param model: fitted model
        @return: dict with metrics
        """

        train_acc = accuracy_score(y_train, y_train_pred > 0.5)
        test_acc = accuracy_score(y_test, y_test_pred > 0.5)

        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred > 0.5)
        test_f1 = f1_score(y_test, y_test_pred > 0.5)

        train_precision = precision_score(y_train, y_train_pred > 0.5)
        test_precision = precision_score(y_test, y_test_pred > 0.5)

        train_recall = recall_score(y_train, y_train_pred > 0.5)
        test_recall = recall_score(y_test, y_test_pred > 0.5)

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,

        }

    def plot_results(self):
        """
        plot error distribution and prediction vs actual scatter plot
        @param model: fitted scikit learn model
        @param model_index: index of model stored in dictionary
        @return: None
        """
        plt.figure(figsize=(10, 10))
        plt.title(f'Predictions in function of actual values for model of experiment {self.experiment_name}')
        sns.scatterplot(x='Actual', y='Prediction', data=pred_actual)
        plt.savefig(f'Data_Processed/scatterplot_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.title(f'Error distribution for model of experiment {self.experiment_name}')
        sns.histplot(x='Error', data=pred_actual)
        plt.savefig(f'Data_Processed/histplot_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_learning_curves(self):

        plt.figure()
        plt.title('Model learning curves')
        # plot learning curves
        plt.plot(results['validation_0'][metric], label=f'train {metric}')
        plt.plot(results['validation_1'][metric], label=f'test {metric}')
        # show the legend
        plt.legend()
        # show the plot
        plt.savefig(f'Data_Processed/learning_curves_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_probabilities(self, model=None, model_index=None, classes=None):

        if model is None:
            model = self.models[model_index]

        preds = model.predict_proba(self.X_test)

        plt.figure(figsize=(10, 7))
        plt.title('Prediction probability distribution')
        for i, j in zip(list(range(preds.shape[1])), classes):
            plt.hist(preds[:, i], label=j, alpha=0.8)
        plt.legend()
        plt.savefig(f'Data_Processed/pred_proba_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_confusion_matrix(self, classes=None):

        conf_matrix = confusion_matrix(self.y_test, preds)

        if classes:
            conf_matrix = conf_matrix / np.sum(conf_matrix)
            conf_matrix = pd.DataFrame(conf_matrix)
            conf_matrix.columns = classes
            conf_matrix.index = classes

        else:
            conf_matrix = conf_matrix / np.sum(conf_matrix)

        plt.figure(figsize=(10, 7))
        # plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(conf_matrix, annot=True,
                    fmt='.2%', annot_kws={"size": 16})  # font size
        plt.savefig(f'Data_Processed/confusion_matrix_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
        plt.show()

    def plot_roc_curve(self, model=None, model_index=None):

        if model is None:
            model = self.models[model_index]

        if self.multiclass:
            probs = model.predict_proba(self.X_test)
            print(probs)

            plt.figure(figsize=(10, 7))
            skplt.metrics.plot_roc(self.y_test, probs)
            plt.savefig(f'Data_Processed/roc_curve_{self.experiment_name}.png', bbox_inches='tight', dpi=1000)
            plt.show()


train_model(model, criterion, optimizer, 20, 1, 32)
