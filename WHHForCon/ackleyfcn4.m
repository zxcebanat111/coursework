function scores = ackleyfcn4(pop)
    [m, n] = size(pop);
    
    scores = zeros(m, 1); 
   
   for i = 1:m
      for j = 1:(n - 1)
            scores = scores + exp(-0.2) .* sqrt( pop(i, j) .^ 2 + pop(i, j + 1) .^ 2 ) ...
            + 3 * ( cos(2 * pop(i, j)) + sin(2 * pop(i, j + 1)) );
      end
   end
end