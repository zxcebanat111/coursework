function pop = ridgefcn(x, d, alpha)

    if nargin < 3 
        alpha = 0.5;
    end
    if nargin < 2
        d = 1;
    end
        
    x1 = x(:, 1);
    pop = x1 + d * (sum(x(:, 2:end).^2, 2) .^ alpha);
end