function pop = qing(x)
    n = size(x, 2);
    x2 = x .^2;
    
    pop = 0;
    for i = 1:n
        pop = pop + (x2(:, i) - i) .^ 2;
    end
end 