function modelstruct = ...
    random_model(num_resources,num_subparts,trans_prior,given_notknow_prior,given_know_prior,pi_0_prior)

import util.dirrnd

% defaults

if ~exist('num_resources','var'), num_resources = 1; end
if ~exist('num_subparts','var'), num_subparts=1; end

if ~exist('trans_prior','var')
    trans_prior = reshape(repmat([20,4;1,20]',1,num_resources),2,2,num_resources);
end
if ~exist('given_notknow_prior','var')
    given_notknow_prior = repmat([5;0.5],1,num_subparts);
end
if ~exist('given_know_prior','var')
    given_know_prior = repmat([0.5;5],1,num_subparts);
end
if ~exist('pi_0_prior','var')
    pi_0_prior = [100;1];
end

As = dirrnd(trans_prior);
given_notknow = dirrnd(given_notknow_prior);
given_know = dirrnd(given_know_prior);
emissions = cat(1,reshape(given_notknow,[1,2,num_subparts]),...
                  reshape(given_know,[1,2,num_subparts]));
pi_0 = dirrnd(pi_0_prior);

modelstruct = struct;
modelstruct.prior = pi_0(2);
modelstruct.learns = As(2,1,:);
modelstruct.forgets = As(1,2,:);
modelstruct.guesses = given_notknow(2,:,:);
modelstruct.slips = given_know(1,:,:);

modelstruct.As = As;
modelstruct.emissions = emissions;
modelstruct.pi_0 = pi_0;

end
