function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%error = (X * theta) -y;
%temp0 = theta(1) - (alpha * 1/m) * (sum(error).*(X(:,1)));
%temp1 = theta(1) - (alpha * 1/m) * (sum(error).*(X(:,2)));
%theta = [temp0, temp1];

%printf('temp0 is \n');
%printf(temp0);

%printf('temp1 is \n');
%printf(temp1);

%theta = theta - (alpha * 1/m) * (X') * (X*theta - y)
  %  h = X * theta;
    
   % hmy = (h-y) .* X(:,1);
   % hmyx =(h-y) .* X(:,2);
   % val = sum(hmy) * (1/m);
   % val1 = sum(hmyx) * (1/m);
   % temp1 = theta(1) - (alpha * val) ;
   % temp2 =  theta(2) - (alpha * val1) ;
   % theta(1) = temp1;
   % theta(2) = temp2;
	
	
	
h = X*theta; % m*n n*1 -> m*1  
diff = h-y; % m*1  
theta_change = (X'*diff)*alpha/m; % n*m × m*1 × n*1 / 1  
theta = theta - theta_change; 

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
