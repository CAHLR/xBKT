function print_dot(num,tot,newline_every)
    if nargin < 3, newline_every = 25; end
    fprintf('.');

    if (mod(num,newline_every) == 0) || (num == tot)
        fprintf(' [%d/%d]\n',num,tot);
    end
end
