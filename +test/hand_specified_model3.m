%% parameters
num_subparts = 1;
num_resources = 2;
num_fit_initializations = 10;
observation_sequence_lengths = 100*ones(1,500);

%% generate synthetic model and data
% model is really easy
truemodel = struct;

truemodel.As = zeros(2,2,num_resources);
for i=1:num_resources
    % truemodel.As(:,:,i) = util.dirrnd(5*[0.9, 0.1; 0.01, 0.99]');
    truemodel.As(:,:,i) = [0.7, 0.3; 0.01, 0.99]';
end
truemodel.learns = truemodel.As(2,1,:); % from state 1 to state 2
truemodel.forgets = truemodel.As(1,2,:);

truemodel.pi_0 = [0.9;0.1];
truemodel.prior = truemodel.pi_0(2);

truemodel.guesses = 0.1*ones(1,num_subparts);
truemodel.slips = 0.03*ones(1,num_subparts);

% data!
disp('generating data...');
data = generate.synthetic_data(truemodel,observation_sequence_lengths);

%% fit models, starting with random initializations
disp('fitting! each dot is a new EM initialization');

best_likelihood = -inf;
% for i=1:num_fit_initializations
%     util.print_dot(i,num_fit_initializations);
%     fitmodel = generate.random_model(num_resources,num_subparts);
%     % fitmodel = truemodel; % NOTE: include this line to initialize at the truth
%     [fitmodel, log_likelihoods] = fit.EM_fit(fitmodel,data);
%     if (log_likelihoods(end) > best_likelihood)
%         best_likelihood = log_likelihoods(end);
%         best_model = fitmodel;
%     end
% end

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

