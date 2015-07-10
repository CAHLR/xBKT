function model = M_step(trans_softcounts,emission_softcounts,init_softcounts)

% avoid NaNs by setting all-zero rows to be uniform
trans_softcounts(:,sum(trans_softcounts,1) == 0) = 1;
emission_softcounts(:,sum(emission_softcounts,2) == 0) = 1;
assert(size(trans_softcounts,1) == 2)
assert(size(trans_softcounts,2) == 2)

model.As = bsxfun(@rdivide,trans_softcounts,sum(trans_softcounts,1));
model.learns = model.As(2,1,:);
model.forgets = model.As(1,2,:);

model.emissions = bsxfun(@rdivide,emission_softcounts,sum(emission_softcounts,2));
model.guesses = squeeze(model.emissions(1,2,:));
model.slips = squeeze(model.emissions(2,1,:));

model.pi_0 = init_softcounts(:) ./ sum(init_softcounts(:));
model.prior = model.pi_0(2);

end
