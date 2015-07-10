function model = resample_params(priors,trans_counts,emission_counts,init_counts,temp)

% TODO could use temp in here too, but the important part is using it to scale
% the likelihoods in sample_states.cpp

outmodel = struct;

model.As = util.dirrnd(trans_counts + priors.As);
model.learns = model.As(2,1,:);
model.forgets = model.As(1,2,:);

new_emission_matrix = util.dirrnd(emission_counts + priors.emissions);
model.guesses = squeeze(new_emission_matrix(1,2,:));
model.slips = squeeze(new_emission_matrix(2,1,:));

model.pi_0 = util.dirrnd(init_counts + priors.inits);
model.prior = model.pi_0(2);

end
