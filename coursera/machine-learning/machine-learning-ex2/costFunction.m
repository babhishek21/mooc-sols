function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X is [m, n+1] matrix
% theta is [n+1, 1] matrix

% h is [m, 1] matrix
h = sigmoid(X * theta);

% y is also a [m, 1 matrix]
% To prevent dimensionality problems, y can be transposed for 
% calculating log likelihood function
logLikelihood = y'*log(h) + (1 - y)'*log(1 - h); % a [1, 1] matrix (scalar)

J = - logLikelihood ./ m;

% grad should also be [n+1, 1] matrix (same dimensions as theta)
grad = (X'*h - X'*y) ./ m;

% =============================================================

end
