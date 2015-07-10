%% parameters
num_subparts = 4;
num_resources = 2;
num_fit_initializations = 25;
observation_sequence_lengths = 100*ones(1,50);

%% generate synthetic model and data
% model is really easy
truemodel = struct;

truemodel.As = cat(3,[0.75, 0.25; 0.1, 0.9]',[0.9, 0.1; 0.1, 0.9]');
truemodel.learns = truemodel.As(2,1,:);
truemodel.forgets = truemodel.As(1,2,:);

truemodel.pi_0 = [0.9;0.1];
truemodel.prior = 0.1;

truemodel.guesses = 0.05 * ones(1,num_subparts);
truemodel.slips = 0.25 * ones(1,num_subparts);

truemodel.resources = randi(num_resources,1,sum(observation_sequence_lengths(:)));

% data!
disp('generating data...');
data = generate.synthetic_data(truemodel,observation_sequence_lengths);

%% fit models, starting with random initializations
disp('fitting! each dot is a new EM initialization');

best_likelihood = -inf;
for i=1:num_fit_initializations
    util.print_dot(i,num_fit_initializations);
    fitmodel = generate.random_model(num_resources,num_subparts);
    % fitmodel = truemodel; % NOTE: include this line to initialize at the truth
    [fitmodel, log_likelihoods] = fit.EM_fit(fitmodel,data);
    if (log_likelihoods(end) > best_likelihood)
        best_likelihood = log_likelihoods(end);
        best_model = fitmodel;
    end
end

%% compare the fit model to the true model

disp('');

disp('these two should look similar');
truemodel.As
best_model.As

disp('these should look similar too');
1-truemodel.guesses
1-best_model.guesses

disp('these should look similar too');
1-truemodel.slips
1-best_model.slips
