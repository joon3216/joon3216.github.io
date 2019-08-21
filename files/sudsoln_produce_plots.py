
import matplotlib.pyplot as plt
import numpy as np
exec(open('sudsoln_produce_results.py').read())

# Scatterplot: time vs. trial
fig, ax = plt.subplots()
for solved_how in ['logically', 'forcefully', 'not_solved']:
    ax.scatter(
        x = result_sudsoln.loc[lambda x: x['solved'] == solved_how].trial,
        y = result_sudsoln.loc[lambda x: x['solved'] == solved_how].time,
        alpha = .2,
        label = solved_how
    )
ax.legend(title = 'How were they solved?')
ax.grid(True)
plt.xlabel('trial')
plt.ylabel('time (in sec)')
plt.title('Scatterplot of time vs. trial')
plt.show()


# Lineplot: prob vs. max_trial
prob_vs_max_trial = {'max_trial': [], 'prob': []}
numer = result_sudsoln.shape[0]
for attempt in range(max_trial + 1):
    denom = result_sudsoln\
        .loc[lambda x: (x['trial'] <= attempt) & (x['is_solved'])]\
        .total\
        .sum()
    prob_vs_max_trial['max_trial'].append(attempt)
    prob_vs_max_trial['prob'].append(denom / numer)
prob_vs_max_trial = pd.DataFrame(prob_vs_max_trial)

# plt.clf()
plt.plot(prob_vs_max_trial.max_trial, prob_vs_max_trial.prob)
plt.ylim(0, 1) 
plt.yticks(np.arange(0, 1.1, .1))
plt.grid(True)
plt.xlabel('max_trial in ss.Sudoku.solve(max_trial)')
plt.ylabel('P(getting solved)')
plt.title('Expected P(getting solved) vs. max_trial')
plt.show()


# Scatterplot: log(time) vs. log(trial + 1)
X = result_sudsoln.loc[:, ['total', 'trial']]
X.trial = np.log(X.trial + 1)
y = result_sudsoln.loc[:, ['time']]
y.time = np.log(y.time)
X, y = X.values, y.values
betas = np.linalg.inv(X.T @ X) @ X.T @ y
model_x = np.linspace(0, np.max(X[:, 1]))
model_y = betas[0] + betas[1] * model_x

# plt.clf()
fig, ax = plt.subplots()
for solved_how in ['logically', 'forcefully', 'not_solved']:
    ax.scatter(
        x = np.log(
            result_sudsoln.loc[lambda x: x['solved'] == solved_how].trial + 1
        ),
        y = np.log(
            result_sudsoln.loc[lambda x: x['solved'] == solved_how].time
        ),
        alpha = .2,
        label = solved_how
    )
ax.legend(title = 'How were they solved?')
ax.grid(True)
plt.plot(model_x, model_y, alpha = .5)
plt.xlabel('log(trial + 1)')
plt.ylabel('log(time)')
plt.title('Scatterplot of log(time) vs. log(trial + 1)')
plt.show()


# time vs. trial according to the fitted model
model_time = lambda x: np.exp(betas[0] + betas[1] * np.log(x + 1))
model_trials = np.arange(201)
model_times = model_time(model_trials)

# plt.clf()
fig, ax = plt.subplots()
for solved_how in ['logically', 'forcefully', 'not_solved']:
    ax.scatter(
        x = result_sudsoln.loc[lambda x: x['solved'] == solved_how].trial,
        y = result_sudsoln.loc[lambda x: x['solved'] == solved_how].time,
        alpha = .2,
        label = solved_how
    )
ax.legend(title = 'How were they solved?')
ax.grid(True)
plt.xlabel('trial')
plt.ylabel('time (in sec)')
plt.title('Scatterplot of time vs. trial with a regression line')
plt.plot(model_trials, model_times, alpha = .5)
plt.show()
