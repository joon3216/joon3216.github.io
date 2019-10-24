
from datetime import datetime as dt

pairs = { ## Selected pairs to compute pairwise interactions
    'V1': ['V2', 'V4', 'V5', 'V10', 'V17', 'V20'],
    'V2': ['V3', 'V7', 'V10', 'V17', 'V20'], 
    'V3': ['V10', 'V20'], 
    'V4': ['V10', 'V17', 'V20', 'V24'], 
    'V5': ['V6', 'V10', 'V20', 'V23'], 
    'V6': ['V10'],
    'V7': ['V10', 'V14', 'V20', 'V21'],
    'V8': ['V10'],
    'V9': ['V10', 'V14', 'V20', 'V25'], 
    'V10': ['V11', 'V20', 'V28'], 
    'V11': ['V12', 'V26'],
    'V12': ['V14', 'V20'],
    'V14': ['V19', 'V20'], 
    'V16': ['V17'],
    'V17': ['V19', 'V20'] 
}
pairs_r = {
    'V1': ['V5', 'V17'],
    'V2': ['V3', 'V10', 'V17'], 
    'V3': ['V10', 'V20'], 
    'V4': ['V10', 'V17', 'V20', 'V24'], 
    'V5': ['V6'],
    'V7': ['V14', 'V21'],
    'V8': ['V10'],
    'V10': ['V28'], 
    'V11': ['V12', 'V26'],
    'V14': ['V20'],
    'V17': ['V19']
}
mod_features = list(cc_train.columns)
int_name = '(Intercept)'

# mod_asis
mod_asis_features = mod_features[:]
mod_asis_features.remove('Class')
mod_asis_features.remove('Time')
mod_asis_features.insert(0, int_name)

# mod_full
mod_full_features = mod_features[:]
mod_full_features.remove('Class')
mod_full_features.remove('Time')
mod_full_features.remove('Amount')
mod_full_features.remove('V13')
mod_full_features.remove('V15')
mod_full_features.append('kde_occu_est')
mod_full_features.append('kde_logAmt_p1')
for k, V in pairs.items():
    for v in V:
        mod_full_features.append('kde_{0}_{1}'.format(k, v))
mod_full_features.insert(0, int_name)

# mod_v
mod_v_features = [ # mod_v features based on cc_train
    int_name, 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
    'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
    'kde_occu_est', 'kde_logAmt_p1', 
    'kde_V1_V5', 'kde_V1_V20', 
    'kde_V2_V3', 
    'kde_V4_V10', 'kde_V4_V24', 
    'kde_V5_V23', 
    'kde_V6_V10', 
    'kde_V7_V21', 
    'kde_V9_V14', 'kde_V9_V25', 
    'kde_V10_V11', 'kde_V10_V28', 
    'kde_V11_V12', 'kde_V11_V26', 
    'kde_V12_V14', 
    'kde_V16_V17', 
    'kde_V17_V19'
]

# mod_r
mod_r_features = [
    int_name, 
    'V3', 'V4', 'V5', 'V7', 'V9', 
    'V11', 'V14', 'V17', 'V19', 'V20', 
    'V24', 'V27', 
    'kde_occu_est', 
    'kde_V1_V5', 'kde_V1_V17', 
    'kde_V2_V3', 'kde_V2_V10', 'kde_V2_V17', 
    'kde_V3_V10', 'kde_V3_V20', 
    'kde_V5_V6', 
    'kde_V7_V14', 'kde_V7_V21', 
    'kde_V8_V10', 
    'kde_V10_V28', 
    'kde_V11_V12', 'kde_V11_V26', 
    'kde_V14_V20', 
    'kde_V17_V19'
]

# Features for ternary classification
ter_features_to_use = [
    int_name, 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
    'kde_occu_est', 'kde_logAmt_p1'
]
ter_features_to_use2 = [
    int_name,
    'V3', 'V4', 'V5', 'V7', 'V9',
    'V11', 'V14', 'V17', 'V19', 'V20', 
    'V24', 'V27',
    'kde_occu_est'
]

# Classification results
results = {
    'fold': [],
    'case': [],
    'model': [],
    'dataset': [],
    'accuracy': [],
    'p_thres': [],
    'class': [],
    'classified': [],
    'class_total': [],
    'counts': [],
    'perc': [],
    'perc_types': []
}
datasettypes = ['train', 'test']
cases = ['vb', 'tb']

# Iterations
start = dt.now()
for fold, train in cc_trains.items():
    start_load = dt.now()
    train = add_intercept(train, int_name, 1)
    test = add_intercept(cc_tests[fold], int_name, 1)

    # Data preparation (both ways produce the same datasets)
    ## Way 1
    # train_kde = design_data(train, train, pairs)
    # test_kde = design_data(test, train, pairs)
    ## Way 2
    computed_kdes = compute_kdes(train, pairs)
    train_kde = design_data(train, kdes = computed_kdes)
    test_kde = design_data(test, kdes = computed_kdes)
    end_load = dt.now()

    start_model = dt.now()
    # Vanilla binary classifiers
    mod_asis =\
        sm.GLM(
            train['Class'],
            train[mod_asis_features],
            family = sm.families.Binomial()
        )\
        .fit()
    mod_full =\
        sm.GLM(
            train_kde['Class'],
            train_kde[mod_full_features],
            family = sm.families.Binomial()
        )\
        .fit()
    mod_v =\
        sm.GLM(
            train_kde['Class'],
            train_kde[mod_v_features],
            family = sm.families.Binomial()
        )\
        .fit()
    mod_r =\
        sm.GLM(
            train_kde['Class'],
            train_kde[mod_r_features],
            family = sm.families.Binomial()
        )\
        .fit()

    # Ternary-binary classifiers
    mod_asis_ter = terbin_model(
        mod_asis, 
        train
    )
    mod_full_ter = terbin_model(
        mod_full, 
        train_kde,
        ter_features = ter_features_to_use
    )
    mod_v_ter = terbin_model(
        mod_v,
        train_kde,
        ter_features = ter_features_to_use
    )
    mod_r_ter = terbin_model(
        mod_r,
        train_kde,
        ter_features = ter_features_to_use2
    )
    end_model = dt.now()

    # Collecting classification results
    mods = {
        'mod_asis': {
            'models': {'vb': mod_asis, 'tb': mod_asis_ter},
            'train': train,
            'test': test
        },
        'mod_full': {
            'models': {'vb': mod_full, 'tb': mod_full_ter},
            'train': train_kde,
            'test': test_kde
        },
        'mod_v': {
            'models': {'vb': mod_v, 'tb': mod_v_ter},
            'train': train_kde,
            'test': test_kde
        },
        'mod_r': {
            'models': {'vb': mod_r, 'tb': mod_r_ter},
            'train': train_kde,
            'test': test_kde
        }
    }
    for mod, info in mods.items():
        for case, dat in [(i, j) for i in cases for j in datasettypes]:
            results['fold'].extend([fold] * 4)
            results['case'].extend([case] * 4)
            results['model'].extend([mod] * 4)
            results['dataset'].extend([dat] * 4)
            case_counts = count_cases(
                info['models'][case],
                info[dat],
                info['train']
            )
            if case == 'vb':
                results['p_thres'].extend([case_counts['p_thres']] * 4)
            else:
                results['p_thres'].extend([
                    case_counts['p_threses'][1], # TN
                    case_counts['p_threses'][2], # FP
                    case_counts['p_threses'][0], # FN
                    case_counts['p_threses'][1]  # TP
                ])
            for c in case_counts['counts'].columns:
                results[c].extend(case_counts['counts'][c])
            results['perc_types'].extend(['tnr', 'fpr', 'fnr', 'tpr'])
            results['accuracy'].extend([case_counts['accuracy']] * 4)

    msg =\
        'Fold {0}:\n' +\
        '    * data transformation time: {1}\n' +\
        '    * modeling time: {2}'
    msg = msg.format(
        fold,
        end_load - start_load,
        end_model - start_model 
    )
    print(msg)
end = dt.now()
print('Total elapsed time: {0}'.format(end - start))

results = pd.DataFrame(results)
