function a = dirrnd(alphavec)

a = gamrnd(alphavec,1);
a = bsxfun(@rdivide,a,sum(a,1));

end
