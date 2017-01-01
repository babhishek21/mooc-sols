function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% size(X)
% X is [m, n+1] matrix

% size(theta)
% theta is a [n+1, 1] matrix

% size(y)
% y is a [m, 1] matrix

% h is a [m, 1] matrix (must have same dimensions as y)
h = X*theta;

sqrErr = (h - y).^2;
J = sum(sqrErr)./(2*m);



% =========================================================================

end
