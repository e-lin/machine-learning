function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate J cost:
H = theta' * X';
R = (H' - y).**2;

temp1 = 0;
for k = 1:m
    temp1 += R(k);
end

n = length(theta);
temp2 = 0;
for t = 2:n     % as j is start from 1, not 0
    temp2 += theta(t)**2;
end

J = (1/(2*m))*temp1 + (lambda/(2*m))*temp2;

% calculate gradient:
G = H' - y;
grad = (1/m)*(G'*X);
for t = 2:n
    grad(t) = grad(t) + (lambda/m)*theta(t);
end
% =========================================================================

grad = grad(:);

end
