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

H = sigmoid(theta' * X'); % size(H) = 1*100
R = -y .* log(H') - (1-y) .* log(1-H');
G = H' - y; %size(G) = 100*1

temp1 = 0;
for k = 1:m
    temp1 += R(k);
end

J = (1/m)*temp1;
grad = (1/m)*(G'*X);

% =============================================================

end
