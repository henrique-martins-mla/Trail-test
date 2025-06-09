import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import shap
import pickle
import pandas as pd

plt.rcParams['figure.figsize'] = 10, 8
plt.rcParams['figure.facecolor'] = 'w'  # remove the default background transparency

from fastprogress import progress_bar

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 100)

from scipy.stats import percentileofscore
from zipfile import ZipFile
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


import mlflow
from trail import Trail


import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def evaluate(y_true_class, y_pred_score, num_in_target=None, cut_prob=None, round_evals=True):
    """
    Apply a batch of classification metrics to evaluate the performance of a
    model that tries to predict the true binary target in `y_true_class`, by
    assigning samples the probabilities in `y_pred_score`.

    For the metrics that require a predicted class, instead of a probability,
    the parameters `cut_prob` and `num_in_target` control the class assignment.
    Samples with a `y_pred_score >= cut_prob` receive a class=1. If the
    probability cutoff `cut_prob` is undefined, the `num_in_target` rows with
    greatest probability receive a class=1. Should `num_in_target` be a value in
    [0, 1], that fraction of samples with greatest probability gets a class=1.

    See also:
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    https://en.wikipedia.org/wiki/Confusion_matrix
    """
    if cut_prob is None:
        if num_in_target is None:
            num_in_target = y_true_class.mean()
        if num_in_target < 1:
            num_in_target = int(round(num_in_target * len(y_pred_score)))
        decr_y_score = y_pred_score.sort_values(ascending=False)
        cut_prob = decr_y_score.iloc[num_in_target - 1]
    
    # apply a probability cutoff to assign the binary classes
    y_pred_class = (y_pred_score >= cut_prob).astype(int)
    num_in_target = y_pred_class.sum()
    
    thr_pct = percentileofscore(y_pred_score, cut_prob) / 100
    neg_pred_mean = y_pred_score[y_pred_score < cut_prob].mean()
    
    
    roc_auc = roc_auc_score(y_true_class, y_pred_score)
    
    # summarizes a precision-recall curve as the weighted mean of precisions
    # achieved at each threshold, with the increase in recall from the previous
    # threshold used as the weight
    ap = average_precision_score(y_true_class, y_pred_score)
    
    
    conf_matrix = confusion_matrix(y_true_class, y_pred_class)
    (tn, fp), (fn, tp) = conf_matrix
    
    assert (tp + fn) == y_true_class.sum() #[0] # samples that truly are in the target
    assert (tp + fp) == num_in_target         # samples model places in the target
    
    # precision or positive predictive value (PPV)
    precision = tp / (tp + fp)
    # in a marketing campaign: it's the expected conversion rate
    # true_positives / num_in_target
    
    # sensitivity, recall, hit rate, or true positive rate (TPR)
    recall = tp / (tp + fn)
    # in a marketing campaign: fraction of clients who converted
    # (positive target) that can be found among the campaign's targets
    # true_positives / total_positives
    
    # specificity, selectivity or true negative rate (TNR)
    # proportion of actual negatives that are correctly identified as such
    specificity = tn / (tn + fp)
    
    # Negative Predictive Value
    # equivalent to precision, but for the prediction of negatives
    NPV = tn / (tn + fn)
    
    # False Omission Rate (= 1 - NPV)
    FOR = fn / (fn + tn)
    FOR_lift = FOR / ((fn + tp) / (fn + tp + tn + fp))
    
    # proportion of true results (both true positives and true negatives)
    # among the total number of cases examined
    accuracy = (tp + tn) / conf_matrix.sum()
    
    # F-score, measure of a test's accuracy. It's the harmonic mean of the
    # precision and recall, where an F1 score reaches its best value at 1
    # (perfect precision and recall) and worst at 0. F_2 variant: weighs
    # recall higher than precision (places more emphasis on false negatives)
    b = 1; F_1 = (1 + b*b) * precision * recall / (b*b * precision + recall)
    b = 2; F_2 = (1 + b*b) * precision * recall / (b*b * precision + recall)
    
    # how many more times are we more likely to find true positives among the
    # samples the model classifies as true, than if we were to pick at random
    precision_lift = precision / y_true_class.mean() #[0]
    
    
    evals = (
        ('roc_auc'           , roc_auc),
        ('average_precision' , ap),
        ('precision_lift'    , precision_lift),
        ('precision'         , precision),
        ('recall'            , recall),
        ('TNR (neg. recall)' , specificity),
        ('NPV (neg. precis.)', NPV),
        ('FOR (default rate)', FOR),
        ('FOR_lift'          , FOR_lift),
        ('F1'                , F_1),
        ('F2'                , F_2),
        ('accuracy'          , accuracy),
        #('confusion_matrix'  , conf_matrix),
        ('tn'                , tn),
        ('fp'                , fp),
        ('fn'                , fn),
        ('tp'                , tp),
        ('base_rate'         , y_true_class.mean()),
        ('class_threshold'   , cut_prob),
        ('nr_in_target tp+fp', tp + fp),
        ('true_targets tp+fn', tp + fn),
        ('threshold_pcentile', thr_pct),
        ('neg_pred_mean'     , neg_pred_mean),
        )
    
    evals = pd.Series(*zip(*[(v, k) for (k, v) in evals]))
    
    return evals.round(4) if round_evals else evals


def best_cutoff(
        y_true_class, y_pred_score, max_metric=None,
        #tn_cost=None, fp_cost=None, fn_cost=None, tp_cost=None
        ):
    """
    Computes the best cut off for the model given the true prediction values
    "y_true_class" and the prediction scores "y_pred_score".
    Optionally, one can define one metric to maximize.
    This function is built on top of the 'evaluate()' function.
    """
    metrics_df = pd.DataFrame([])

    with np.errstate(invalid='ignore'):
        for i in np.linspace(0, 1, 101):
            metrics = evaluate(
                y_true_class, y_pred_score, cut_prob=i,
                #tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, tp_cost=tp_cost
                round_evals=False)
            
            metrics_df = pd.concat([metrics_df, metrics], axis=1)

    metrics_df = metrics_df.T
    metrics_df.index = np.linspace(0, 1, 101)

    if max_metric == None:
        return metrics_df
    else:
        return metrics_df[metrics_df[max_metric] == metrics_df[max_metric].max()].T



def null_stats(df):
    """
    Statistics on the number and percentage of missing values.
    """
    df = df.to_frame() if isinstance(df, pd.Series) else df
    isnull_sum = df.isnull().sum()
    isnull_pct = isnull_sum / len(df) * 100

    isnull_sum.name = '# NaN'
    isnull_pct.name = '% NaN'

    return pd.concat([isnull_sum, isnull_pct], axis=1)


def load(fname, mode='rb', open=open):
    with open(fname, mode) as f:
        d = pickle.load(f)
    return d


def encode(data_file, data_dict_file):
    d = pd.read_excel(data_dict_file)
    index_cols = list(d['feature'][d['feature_type'] == 'index'])
    input_cols = list(d['feature'][d['feature_type'] == 'input'])
    target_col = list(d['feature'][d['feature_type'] == 'target'])
    categ_cols = list(d['feature'][d['data_type'] == 'category'])
    
    X = pd.read_csv(data_file)
    if index_cols != []: X.set_index(index_cols, inplace=True)
    
    assert len(target_col) == 1
    y = X[target_col[0]]
    
    X = X[input_cols] # enforce column ordering
    X[categ_cols] = X[categ_cols].astype('category')
    
    return X, y


def predict(model, X):
    if isinstance(model, str):
        model = load(model)
    
    y = pd.DataFrame(model.predict_proba(X)).iloc[:,1]
    
    if isinstance(X, pd.DataFrame):
        y.index = X.index
    
    return y



with ZipFile('outputs/ls 01/dataset/GiveMeSomeCredit.zip', 'r') as zf:
    
    with zf.open('cs-training.csv', mode='r') as f:
        dataset_raw = pd.read_csv(f)
        
    with zf.open('Data Dictionary.xls', mode='r') as f:
        data_dict_raw = pd.read_excel(f, header=1)


# Mapping adapted from:
# https://gitlab.mlanalytics.pt/ml-analytics/mvp/research/-/blob/705fd2d6e183be097400bc036d77177cd04c1b7a/src/mvp/dataset.py#L34

# transform from CamelCase to Snake case
column_mapping = {
    'MonthlyIncome'                        : 'monthly_income',
    'DebtRatio'                            : 'debt_ratio',
    'NumberOfDependents'                   : 'nr_dependents',
    'RevolvingUtilizationOfUnsecuredLines' : 'credit_balances_dividedby_limits',
    'NumberOfOpenCreditLinesAndLoans'      : 'nr_open_credit_lines_and_loans',
    'NumberRealEstateLoansOrLines'         : 'nr_real_estate_loans',
    'NumberOfTime30-59DaysPastDueNotWorse' : 'nr_times_30_59_days_past_due',
    'NumberOfTime60-89DaysPastDueNotWorse' : 'nr_times_60_89_days_past_due',
    'NumberOfTimes90DaysLate'              : 'nr_times_90plus_days_past_due',
    'SeriousDlqin2yrs'                     : 'serious_delinquency',
    }


data_dict = data_dict_raw.copy()

# rename the variable names, and enforce the desired ordering
data_dict['Variable Name'] = data_dict['Variable Name'].map(column_mapping)
data_dict = data_dict.set_index('Variable Name').loc[list(column_mapping.values())]

# add the `feature_type` column, with the format expected by the `encode()` function above
data_dict['feature_type'] = 'input'
data_dict.loc['serious_delinquency', 'feature_type'] = 'target'

data_dict.index.name = 'feature'
data_dict.columns = ['description', 'data_type', 'feature_type']

# drop index column [it's just a 1-based sequential indexing]
dataset = dataset_raw.drop(columns='Unnamed: 0')

# rename columns, and enforce the desired ordering
dataset = dataset.rename(columns=column_mapping)[list(column_mapping.values())]


data_path      = r'outputs/ls 01/'
data_file      = data_path + 'GiveMeSomeCredit__dataset.csv'
data_dict_file = data_path + 'GiveMeSomeCredit__data_dictionary.xlsx'
model_file     = data_path + 'GiveMeSomeCredit__model.pkl'

X, y_true = encode(data_file, data_dict_file)
X.fillna(-1, inplace=True)


def train(lr=0.3, n_estimators=20, reg_lambda=1.0, run_name="custom_run"):

    mlflow.lightgbm.autolog()

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        run.data.tags["mlflow.source.git.commit"] = sha
        with Trail(experiment_title=run_name):
            model = LGBMClassifier(learning_rate=lr, n_estimators=n_estimators, reg_lambda=reg_lambda, random_state=42, n_jobs=-1)
            # Train set metrics
            model.fit(X, y_true, verbose=10)


            # region explainability with shap
            shap_path = "model_explanations_shap"
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            ensured_list_shap_values = [shap_values[:,:,i] for i in range(shap_values.shape[-1])]

            for i, name in enumerate(X.columns):
                for j in range(shap_values.shape[-1]):
                    shap.dependence_plot(name, shap_values[:,:,j], X)
                    mlflow.log_figure(pl.gcf(), f"{shap_path}/dependence_plot_{name}_class{j}.png", save_kwargs={"bbox_inches": 'tight'})
                    pl.close()

            for i in range(shap_values.shape[-1]):
                shap.summary_plot(ensured_list_shap_values[i], X, plot_type="bar", feature_names = X.columns)
                mlflow.log_figure(pl.gcf(), f"{shap_path}/summary_plot_class{i}.png", save_kwargs={"bbox_inches": 'tight'})
                pl.close()
                # shap.waterfall_plot(explainer(X)[0, :, i])
                shap.plots._waterfall.waterfall_legacy(explainer.expected_value[i], shap_values[0,:,i], feature_names = X.columns)
                mlflow.log_figure(pl.gcf(), f"{shap_path}/waterfall_plot_class{i}_prediction_0.png", save_kwargs={"bbox_inches": 'tight'})
                pl.close()
                shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0], X.iloc[0,:], matplotlib=True, show=False)
                mlflow.log_figure(pl.gcf(), f"{shap_path}/force_plot_class{i}_prediction_0.png", save_kwargs={"bbox_inches": 'tight'})
                pl.close()

            shap.summary_plot(ensured_list_shap_values, X, plot_type="bar", class_names=['Class 0', 'Class 1', 'Class 2'], feature_names = X.columns)
            mlflow.log_figure(pl.gcf(), f"{shap_path}/summary_plot_multiclass.png", save_kwargs={"bbox_inches": 'tight'})
            pl.close()

            # not possible to log for TreeExplainer, only shap.Explainer. Save as pickle instead
            # signature = infer_signature(X, model.predict(X)) 
            # mlflow.shap.log_explainer(explainer, artifact_path=shap_path, signature=signature)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "tree_explainer.pkl")
                
                # Save the explainer to the temp file
                with open(temp_file_path, "wb") as f:
                    pickle.dump(explainer, f)
                mlflow.log_artifact(temp_file_path, artifact_path=shap_path)
            #endregion explainability with shap


            yp = model.predict_proba(y_true)[:,1]
            ev = evaluate(y_true, pd.Series(yp), num_in_target=y_true.mean()); #ev

            for metric, value in ev.tolist():
                mlflow.log_metric(metric, value)

            # register model
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "staging.trail")



run_name = "run_1"
mlflow.lightgbm.autolog()
with mlflow.start_run(run_name=run_name) as run:
    print(run.data.tags.get("mlflow.source.git.commit"))
    with Trail(experiment_title=run_name):
        model = LGBMClassifier(learning_rate=0.3, n_estimators=20, reg_lambda=1.0, random_state=42, n_jobs=-1)
        # Train set metrics
        model.fit(X, y_true, verbose=10)

print(model.__dict__)


# train(run_name="run_1")
# train(0.1, 20, 0.5, run_name="run_2")
# train(0.9, 30, 0.8, run_name="run_3")
# train(0.6, 40, 0.2, run_name="run_4")