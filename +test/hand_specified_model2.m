%% parameters
num_subparts = 1;
num_resources = 2;
num_fit_initializations = 1;
observation_sequence_lengths = 2000*ones(1,50000);

%% generate synthetic model and data
% model is really easy
truemodel = struct;

truemodel.As = cat(3,[0.75, 0.25; 0.1, 0.9]',[0.9, 0.1; 0.1, 0.9]');
truemodel.As(1,2,:) = 0;
truemodel.As(2,2,:) = 1;
truemodel.learns = truemodel.As(2,1,:);
truemodel.forgets = truemodel.As(1,2,:);

truemodel.pi_0 = [0.9;0.3];
truemodel.prior = 0.3;

truemodel.guesses = [0.10];
truemodel.slips = [0.03];

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

fitmodel = truemodel; % NOTE: include this line to initialize at the truth
[fitmodel, log_likelihoods] = fit.EM_fit(fitmodel,data);
if (log_likelihoods(end) > best_likelihood)
     best_likelihood = log_likelihoods(end);
     best_model = fitmodel;
end


%% compare the fit model to the true model

disp('');
fprintf('\ttruth\tlearned\n');
for r=1:num_resources
	fprintf('learn%d\t%.4f\t%.4f\n',r,squeeze(truemodel.As(2,1,r)),squeeze(best_model.As(2,1,r)));
end
for r=1:num_resources
        fprintf('forget%d\t%.4f\t%.4f\n',r,squeeze(truemodel.As(1,2,r)),squeeze(best_model.As(1,2,r)));
end

for s=1:num_subparts
        fprintf('guess%d\t%.4f\t%.4f\n',s,truemodel.guesses(s),best_model.guesses(s));
end

for s=1:num_subparts
        fprintf('slip%d\t%.4f\t%.4f\n',s,truemodel.slips(s),best_model.slips(s));
end

