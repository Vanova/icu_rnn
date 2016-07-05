import numpy as np
import random
import m2m_rnn as m2m

# one example is one audio file
seq_time_steps = 20
nb_examples = 401 // seq_time_steps
feat_dim = 19
hidden_dim = 100
nb_clcs = 7

X = []
Y = []
for i in xrange(nb_examples):
    X.append(np.random.uniform(-np.sqrt(1./feat_dim), np.sqrt(1./feat_dim), (seq_time_steps, feat_dim)))
    Y.append(np.random.binomial(1, 0.4, nb_clcs))
# to 2D array
Y = np.asarray(Y)
#####

iters=100
eta=0.01
alpha=0.0
lambda2=0.0
dropout=.5
running_total = 0

config = { 'nb_hidden': [hidden_dim, hidden_dim], 'nb_epochs': 10,
            'alpha': 0.0, 'lambda2': 0.0,
            'clip_at': 0.0, 'scale_norm': 0.0,
            'starting_eta': 32.0, 'minimum_eta': 1.0,
            'half_eta_every': 10 }
# config.update(model_config)
P = config['nb_input'] = X[0].shape[1]
K = config['nb_output'] = Y.shape[1]
config['results_dir'] = './'

print 'LSTM Model Configuration\n----------'
for k in sorted(config):
    print k, ':', config[k]
print '----------', '\n'

nb_hidden = config['nb_hidden']
nb_epochs = config['nb_epochs']
eta = config['starting_eta']
min_eta = config['minimum_eta']
half_every = config['half_eta_every']
alpha = config['alpha']
lambda2 = config['lambda2']
clip_at = config['clip_at']
scale_norm = config['scale_norm']

rnn = m2m.M2M_RNN(num_input=P, num_hidden=nb_hidden, num_output=K, clip_at=clip_at, scale_norm=scale_norm)

for it in xrange(iters):
    if it == 27:
        m = 0
    i = random.randint(0, len(X)-1)

    if X[i].shape[0] < 200:
        cost, last_step_cost = rnn.train(X[i], np.tile(Y[i], (len(X[i]), 1)), eta, alpha, lambda2, dropout)

    else:
        cost, last_step_cost = rnn.train(X[i][-200:], np.tile(Y[i], (200, 1)), eta, alpha, lambda2)

    running_total += last_step_cost
    running_avg = running_total / (it + 1.)
    print "iteration: %s, cost: %s, last: %s, avg: %s" % (it, cost, last_step_cost, running_avg)

