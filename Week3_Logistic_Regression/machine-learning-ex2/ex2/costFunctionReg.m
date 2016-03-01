function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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

H = sigmoid(theta' * X'); % size(H) = 1*100
R = -y .* log(H') - (1-y) .* log(1-H');
G = H' - y; % size(G) = 100*1

temp1 = 0;
for k = 1:m
    temp1 += R(k);
end

n = length(theta); % n = 28
temp2 = 0;
for t = 2:n     % as j is start from 1, not 0
    temp2 += theta(t)**2;
end

J = (1/m)*temp1 + (lambda/(2*m))*temp2;

grad = (1/m)*(G'*X); % size(grad) = 1*28
for t = 2:n
    grad(t) = grad(t) + (lambda/m)*theta(t);
end

% =============================================================

end
