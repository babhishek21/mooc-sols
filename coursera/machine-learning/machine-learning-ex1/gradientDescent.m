function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
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

    % delta needs to be a [n+1, 1] matrix
    % delta needs to be (1/m) * sum((h(X[i]) - y[i])*X[i]) where
    % y is [m, 1] matrix and X is [m, n+1] matrix. To fix
    % dimensionality problems, we can instead write delta as
    % delta = (1/m) * ((h(X) - y)' * X)'
    % delta is the actually the gradient of the cost function J

    delta = (X'*X*theta - X'*y)./m;
    theta = theta - (alpha.*delta);

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
