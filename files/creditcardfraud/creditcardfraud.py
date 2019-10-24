
from matplotlib.offsetbox import HPacker
from patsy import dmatrix, dmatrices
from scipy.optimize import curve_fit
from scipy.stats import chi2
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# Functions
exec(open('creditcardfraud_functions.py').read())


# Reordering columns
creditcard = pd.read_csv('creditcard.csv')
Vs = list(map(lambda i: 'V' + str(i), range(1, 29)))
cols =  ['Class', 'Time', 'Amount']
cols.extend(Vs)
creditcard = creditcard.loc[:, cols]


# trains and tests
n_splits = 4
splits = list(range(
    86400, # minimum size of the training set
    172800 + int((24 / n_splits) * 3600), 
    int((24 / n_splits) * 3600)
))
splits = [splits[i:(i + 2)] for i in range(len(splits) - 1)]
cc_trains = {}
cc_tests = {}
for i, row in enumerate(splits):
    cc_trains[i + 1] = creditcard.query('Time <= {0}'.format(row[0]))
    cc_tests[i + 1] =\
        creditcard.query('{0} < Time <= {1}'.format(row[0], row[1]))
cc_train = cc_trains[1]


# Figure: description of each fold
plt.clf()
plt.ylabel('Fold')
plt.xlabel('Hours')
splits_hrs = map(lambda row: [row[0] / 3600, row[1] / 3600], splits)
for fold, hr in enumerate(splits_hrs):
    ind = hr[0] + 1
    zipped = zip(
        [np.arange(ind - 1, hr[1] + 1), np.arange(ind)], 
        ['#FF7F0E', '#1F77B4'], 
        ['test', 'train']
    )
    for rt, col, lab in zipped:
        plt.plot(
            rt, np.zeros(len(rt)) - fold - 1, 
            color = col, linewidth = 3, label = lab
        )
plt.xticks(np.arange(49, step = 24 / n_splits))
plt.annotate('↑cc_train', xy = (12, -1.05), ha = 'center', va = 'top')
lst_train_keys = np.arange(-n_splits, 0)
plt.yticks(lst_train_keys, tuple(map(lambda x: str(x), -lst_train_keys)))
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index = True)
labels = [labels[1], labels[0]]
ids = [1, 0]
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc = 'upper right')
plt.subplots_adjust(bottom = .2)
sns.despine(left = True)
plt.grid()
plt.show()


# Checks
print0(
    'Checking creditcard has no missing data:\n',
    pd.DataFrame.equals(creditcard, creditcard.dropna())
)
Time = creditcard['Time']
lst_Time = np.array(Time)
lst_Time_sorted = lst_Time.copy()
lst_Time_sorted.sort()
print0(
    'Checking creditcard is an ordered set:\n', 
    (lst_Time == lst_Time_sorted).all()
)
print0(
    'Checking creditcard.Time is not unique:\n',
    not (Time.value_counts() == 1).all()
)


# Figure: number of transactions at each Time in cc_train
splits_lst = range(
    0, 
    172800 + int((24 / n_splits) * 3600), 
    int((24 / n_splits) * 3600)
)
cc_train_all_Times = get_occurrence(cc_train)
cc_train_fitted_sin = fft_curve(
    cc_train_all_Times['Time'], 
    cc_train_all_Times['Occurrence'], 
    True
)
fitted_func = lambda t: cc_train_fitted_sin['fitfunc'](t)
plt.clf()
plt.plot(
    cc_train_all_Times['Time'], cc_train_all_Times['Occurrence'], 
    alpha = .5,
    label = 'Actual occurrence'
)
plt.plot(
    cc_train_all_Times['Time'], fitted_func(cc_train_all_Times['Time']),
    label = 'Estimated occurrence based on cc_train'
)
plt.legend(loc = 'upper left')
plt.xlabel('Time (in sec)')
plt.xticks(splits_lst[:(splits_lst.index(86400) + 1)])
plt.ylabel('Occurrence')
plt.title('Number of transactions as Time progresses', loc = 'left')
plt.show()


# Figure: occurrence estimates of each training set
plt.clf()
creditcard_all_Times = get_occurrence(creditcard)
plt.plot(
    creditcard_all_Times['Time'], 
    creditcard_all_Times['Occurrence'],
    alpha = .5,
    label = 'Actual occurrence'
)
for fold, train in cc_trains.items():
    train_all_Times = get_occurrence(train)
    train_fitted_sin = fft_curve(
        train_all_Times['Time'], 
        train_all_Times['Occurrence'], 
        True
    )
    train_fitted_func = lambda t: train_fitted_sin['fitfunc'](t)
    fold_label = 'Fold ' + str(fold)
    fold_label += ' (cc_train)' if fold == 1 else ''
    plt.plot(
        creditcard_all_Times['Time'], 
        train_fitted_func(creditcard_all_Times['Time']),
        alpha = .5,
        label = fold_label
    )
plt.legend(loc = 'upper left')
plt.xlabel('Time (in sec)')
plt.xticks(splits_lst)
plt.ylabel('Occurrence')
plt.title('Comparison of different occurrence estimates', loc = 'left')
plt.show()


# Figure: distributions of the Occurrence estimates
all_Times2 =\
    Pipe(cc_train)\
    .pipe(get_occurrence)\
    .pipe(
        pd.merge,
        cc_train[['Time', 'Class']],
        'left',
        'Time'
    )\
    .pipe(pd.DataFrame.dropna)\
    .collect()
plt.clf()
for j in range(2):    
    sns.kdeplot(
        fitted_func(all_Times2.query('Class == ' + str(j))['Time']), 
        bw = .075, 
        label = 'Class ' + str(j)
    )
plt.title(
    'Distributions of the Occurrence estimates', 
    loc = 'left'
)
plt.legend(loc = 'upper left')
plt.show()


# Figure: number of fraudulent transactions at a Time
cc_train_Class1_Time_counts = add_intercept(cc_train)\
    .loc[:, ['Class', 'Time', 'Intercept']]\
    .query('Class == 1')\
    .groupby('Time')\
    .agg(counts = ('Intercept', 'sum'))\
    ['counts']\
    .value_counts()
print0(
    'Number of fraudulent transactions at a Time in cc_train:\n',
    cc_train_Class1_Time_counts
)
plt.clf()
plt.bar(
    [str(i) for i in cc_train_Class1_Time_counts.index], 
    cc_train_Class1_Time_counts
)
for i in cc_train_Class1_Time_counts.index:
    plt.annotate(
        str(cc_train_Class1_Time_counts.loc[i]), 
        xy = (
            str(i), 
            cc_train_Class1_Time_counts.loc[i]
        ), 
        ha = 'center',
        va = 'bottom'
    )
plt.title(
    'Number of fraudulent transactions at a given point in Time',
    loc = 'left'
)
plt.xlabel('Number of fraudulent transactions at a Time')
plt.ylabel('Number of moments')
plt.show()


# Table: description of Amount, and the case count of Amount 0 in cc_train
cc_train_Class_Amount = cc_train[['Class', 'Amount']]
cc_train_Class_Amount_describe = cc_train_Class_Amount\
    .groupby('Class')\
    .describe()
print0(
    'Description of Amount for each Class in cc_train:\n',
    cc_train_Class_Amount_describe
)
cc_train_Class_Amount0_size = cc_train_Class_Amount\
    .query('Amount == 0')\
    .groupby('Class')\
    .size()
print0(
    'Number of cases in each Class where Amount is 0:\n',
    cc_train_Class_Amount0_size
)
cc_train_Class_Amount0_size0 = cc_train_Class_Amount0_size[0]
cc_train_Class_Amount_size0 = cc_train_Class_Amount_describe.iloc[0, 0]
cc_train_Class_Amount0_size1 = cc_train_Class_Amount0_size[1]
cc_train_Class_Amount_size1 = cc_train_Class_Amount_describe.iloc[1, 0]
prop_amt0_class0 = cc_train_Class_Amount0_size0 /\
    cc_train_Class_Amount_size0
prop_amt0_class1 = cc_train_Class_Amount0_size1 /\
    cc_train_Class_Amount_size1


# Figure: distribution of log(Amount + 1) for each Class
cc_train_Class_Amount['logAmt_plus1'] =\
    np.log(cc_train_Class_Amount['Amount'] + 1)
plt.clf()
for i in range(2):
    sns.kdeplot(
        cc_train_Class_Amount\
            .query('Class == ' + str(i))\
            ['logAmt_plus1'],
        label = 'Class ' + str(i),
        bw = .1
    )
plt.title(
    'Distribution of log(Amount + 1) in each Class', 
    loc = 'left'
)
plt.show()


# Investigate modes of Amount
modes = np.zeros(3)
for i in range(2):
    modes[i + 1] += hsm(
        cc_train_Class_Amount\
            .query('Class == 1 & logAmt_plus1 > ' + str(i))\
            ['logAmt_plus1']
    )
print0('Modes of log(Amount + 1) when Class is 1:\n', modes)
print0('Modes of Amount when Class is 1:\n', np.exp(modes) - 1)
lsts = [[1], [99.99], [0, 1, 99.99]]
msg = 'The probability of a transaction being fraud given that:\n'
for lst in lsts:
    cc_train_Class_Amount0199_sizes = cc_train_Class_Amount\
        .loc[lambda x: x['Amount'].isin(lst)]\
        .groupby('Class')\
        .describe()\
        .iloc[:, 0]
    msg += '    * the Amount is {0}: {1}%\n'.format(
        lst,
        round(
            cc_train_Class_Amount0199_sizes[1] /\
                np.sum(cc_train_Class_Amount0199_sizes) * 100, 
            3
        )
    )
print0(msg[:-1])
Pipe(
    cc_train_Class_Amount\
        .query('Amount in [0, 1, 99.99] & Class == 1')\
        .groupby('Amount')\
        .agg(counts = ('Class', 'sum'))\
        .assign(total = cc_train_Class_Amount_size1)
)\
.pipe(mutate, 'perc', lambd_df = lambda df: df['counts'] / df['total'])\
.pipe(lambda df: print0('Estimated P(Amount | fraudulent):\n', df))
Pipe(
    add_intercept(cc_train_Class_Amount, 'one', loc = -1)\
        .loc[lambda x: x['Amount'].isin([0, 1, 99.99])]\
        .groupby(['Class', 'Amount'])\
        .agg(counts = ('one', 'sum'))\
        .reset_index()
)\
.pipe(dcast, 'Class ~ Amount')\
.pipe(lambda df: print0('Counts of (Amount, Class) in cc_train:\n', df))


# Check PCs
for dt, dtn in zip([creditcard, cc_train], [' creditcard ', ' cc_train ']):
    print0(
        'Checking principal components in' + dtn + 'are centered:\n',
        np.isclose(
            dt.loc[:, Vs].describe().loc['mean', :].values, 
            np.zeros(28)
        )\
        .all()
    )
    print0(
        'Checking principal components in' + dtn + 'are uncorrelated:\n',
        np.isclose(
            dt.loc[:, Vs].corr().values.sum(axis = 1) -\
                dt.loc[:, Vs].corr().values.diagonal(),
            np.zeros(28)
        )\
        .all()
    )


# Figure: variances in features explained by each PC
varexp = creditcard.loc[:, Vs].cov().values.diagonal()
perc_varexp = varexp / np.sum(varexp) * 100
cumsum_varexp = np.cumsum(perc_varexp)
PCs = np.arange(1, 29)
plt.clf()
plt.xlim(0, 29)
plt.plot(PCs, cumsum_varexp, label = 'Cumulative proportion')
plt.plot(PCs, perc_varexp, label = 'Proportion of each PC')
plt.title('Variances in features explained by each PC', loc = 'left')
plt.xlabel('Order of PC')
plt.ylabel('Proportion')
plt.xticks(PCs, rotation = 45)
plt.yticks(
    np.arange(110, step = 10), 
    tuple(map(lambda x: str(x) + '%', np.arange(110, step = 10)))
)
plt.grid()
plt.legend()
plt.subplots_adjust(bottom = .15)
plt.show()


# Figure: distribution of each Class in each PC of cc_train
cols = ['Class']
cols.extend(Vs)
plt.clf()
g = sns.FacetGrid(
    data = cc_train\
        .loc[:, cols]\
        .melt(id_vars = 'Class'),
    col = 'variable',
    col_wrap = 4,
    hue = 'Class',
    sharex = False,
    sharey = False
)
g = g.map(sns.kdeplot, 'value', bw = .1).add_legend()
g.despine(left = True)
plt.subplots_adjust(top = .9)
g._legend.set_bbox_to_anchor(bbox = (.525, .945))
g.fig.suptitle(
    'Distribution of each Class in each PC of the Fold 1 training data'
)
plt.show()


# Figure: relationships among PCs
V_select_vs_others('V1')
V_x = list(range(1, 28))
V_x.remove(13); V_x.remove(15)
for i in V_x:
    V_y = list(range(i + 1, 29))
    if 13 in V_y:
        V_y.remove(13)
    if 15 in V_y:
        V_y.remove(15)
    for j in V_y:
        plt.clf()
        V_select_vs_others('V' + str(i), 'V' + str(j))
        sns.kdeplot(
            cc_train.query('Class == 1')['V' + str(i)], 
            cc_train.query('Class == 1')['V' + str(j)]
        )
        plt.show()
V_select_vs_others('V1', 'V2')
V_select_vs_others('V4', 'V16')
V_select_vs_others('V1', ['V16', 'V17'], mfrow = (1, 2))
V_select_vs_others(['V17', 'V18'], 'V19', mfrow = (1, 2))


# Models
exec(open('creditcardfraud_models.py').read())


# Figure: Accuracy/TNR/TPR as Fold changes for each classifier
cols_req = ['fold','case','model','dataset','accuracy','perc','perc_types']
perc_types = ['accuracy', 'tnr', 'tpr']
fig = comparison('case')
plt.subplots_adjust(top = .85)
fig.set_xlabels('Fold')
fig.set_ylabels('Rate')
fig.fig.suptitle(
    'Rate changes for different folds and classifiers',
    fontsize = 15,
    x = .48
)
fig._legend.set_bbox_to_anchor(bbox = (.55, .915))
fig._legend.set_title('')
to_replace = fig._legend.get_children()[0]\
    .get_children()[1]\
    .get_children()[0]
fig._legend.get_children()[0].get_children()[1].get_children()[0] =\
    HPacker(
        pad = to_replace.pad,
        sep = to_replace.sep, 
        children = to_replace.get_children()
    )
plt.show()


# Performance tables
dcast_case_dataset('vb', 'train')
dcast_case_dataset('vb', 'test')
dcast_case_dataset('tb', 'train')
dcast_case_dataset('tb', 'test')


# Figure: Accuracy/TNR/TPR for each model
fig = comparison('model')
plt.subplots_adjust(top = .85)
fig.set_xlabels('Fold')
fig.set_ylabels('Rate')
fig.fig.suptitle(
    'Rate changes for different folds and models',
    fontsize = 15,
    x = .49
)
fig._legend.set_bbox_to_anchor(bbox = (.58, .915))
fig._legend.set_title('')
to_replace = fig._legend.get_children()[0]\
    .get_children()[1]\
    .get_children()[0]
fig._legend.get_children()[0].get_children()[1].get_children()[0] =\
    HPacker(
        pad = to_replace.pad,
        sep = to_replace.sep, 
        children = to_replace.get_children()
    )
plt.show()


# LRTs: mod_r vs. mod_full and mod_v vs. mod_full
print0('mod_r vs. mod_full:\n', anova(mod_r, mod_full))
print0('mod_v vs. mod_full:\n', anova(mod_v, mod_full))


# Aside: Figure: accuracy/tnr/tpr for each classifier-model combo
fig = sns.catplot(
    data = Pipe(results)\
        .pipe(
            mutate, 
            'case_model', 
            lambd_df = lambda df: df['model'] + '_' + df['case']
        )\
        .pipe(
            layers,
            ['fold','case_model','dataset','accuracy','perc','perc_types'],
            perc_types
        )\
        .collect()\
        .sort_values(
            ['fold', 'case_model', 'dataset'],
            ascending = [True, False, False]
        ),
    x = 'fold',
    y = 'perc',
    row = 'dataset',
    col = 'perc_types',
    hue = 'case_model',
    kind = 'point'
)


# Table: mean accuracy/tpr in test sets for each classifier-model combo
print(dcast(
    Pipe(results)\
        .pipe(layers, cols_req, perc_types)\
        .pipe(
            mutate, 
            'classifier', 
            ('case', lambda x: 'Binary' if x == 'vb' else 'Ternary-binary')
        )\
        .collect()\
        .query('dataset == "test"')\
        .query('perc_types != "True negative rate"')\
        .groupby(['classifier', 'model', 'perc_types'])\
        .agg(perc_mean = ('perc', 'mean'))\
        .reset_index(), 
    'classifier + model ~ perc_types'
))


# Figure: observed proportion vs. predicted probability
fig = plt.figure()
mod_names = ['mod_asis', 'mod_full', 'mod_r', 'mod_v']
mods = [mod_asis, mod_full, mod_r, mod_v]
i = 0
for mod_name, mod in zip(mod_names, mods):
    i += 1
    plt.subplot(2, 2, i)
    plot_op(mod, train_kde['Class'], xlab = '', ylab = '')
    plt.title(mod_name)
fig.axes[0].axes.set_xticklabels([])
fig.axes[0].axes.set_ylabel('Observed Proportion')
fig.axes[1].axes.set_xticklabels([])
fig.axes[1].axes.set_yticklabels([])
fig.axes[2].axes.set_xlabel('Predicted Probability')
fig.axes[2].axes.set_ylabel('Observed Proportion')
fig.axes[3].axes.set_xlabel('Predicted Probability')
fig.axes[3].axes.set_yticklabels([])
fig.subplots_adjust(hspace = .2, wspace = .05)
plt.show()


# Table: expected classification rates for the next 6 hours given that
#     the training set is the entire 48 hours and the classifier is vb/tb
expected_rates('vb')[['model', 'accuracy', 'tpr']]
expected_rates('tb')[['model', 'accuracy', 'tpr']]


# Table: logarithmic scoring of models
print(pd.DataFrame({
    'model': mod_names,
    'logarithmic_scoring': map(
        logarithmic_scoring, 
        mods, 
        [add_intercept(cc_tests[1], '(Intercept)'), 
         test_kde, test_kde, test_kde]
    )
}))
