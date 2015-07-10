num_resources = 2;
num_subparts = 1;

truemodel = generate.random_model(num_resources,num_subparts);

observation_sequence_lengths = 100*ones(1,100);
data = generate.synthetic_data(truemodel,observation_sequence_lengths);

trans_softcounts = zeros(2,2,num_resources);
emission_softcounts = zeros(2,2,num_subparts);
init_softcounts = zeros(2,1);

% new E_step
[l,a,g] = fit.E_step(data,truemodel,trans_softcounts,emission_softcounts,init_softcounts);
% old E_step
[l2,al,bl] = fit.E_step_old(data,truemodel,trans_softcounts,emission_softcounts,init_softcounts);

abs(l - l2) < 1

% normalize alphas and compare
a2 = exp(al);
a2 = bsxfun(@rdivide,a2,sum(a2,1));
all(all(abs(a - a2) < 1e-6))

% compute gammas and compare
g2 = exp(al+bl);
g2 = bsxfun(@rdivide,g2,sum(g2,1));
all(all(abs(g - g2) < 1e-6))

