function indices = sample_discrete(scores)

cscores = cumsum(scores,1);
indices = 1+sum(bsxfun(@gt,cscores(end,:) .* rand(1,size(scores,2)),cscores),1);

end
