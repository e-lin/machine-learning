function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ------------------------ Part 1 ------------------------
% --------------------------------------------------------
% Add ones to the X data matrix
A1 = [ones(m, 1) X];

Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

% Add ones to the A2 data matrix
A2 = [ones(m, 1) A2];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

H = A3; % size(H) = 5000*10
R = zeros(m,1);
for n = 1:num_labels
    TMP = H(:,n);
    R += -(y==n) .* log(TMP) - (1-(y==n)) .* log(1-TMP);
end

temp1 = 0;
for k = 1:m
    temp1 += R(k);
end

% J = (1/m)*temp1; % no regularization

% ------------------------ Part 2 ------------------------
% --------------------------------------------------------
E3 = zeros(size(A3));
for n = 1:num_labels
    E3(:,n) = A3(:,n) - (y==n);
end

E2 = Theta2'(2:end,:) * E3';  % size(Theta2'(2:end,:)) = 25*10 as we skip delta_0_(2)
E2 = E2' .* sigmoidGradient(Z2);  % finally, size(D2) = 5000*25

% no regularization
Theta2_grad = 1/m * (E3' * A2);  % 25 * 401
Theta1_grad = 1/m * (E2' * A1);  % 10 * 26

% regularization
Theta2_tmp1 = 1/m * (E3' * A2);
Theta2_tmp1 = Theta2_tmp1(:, 2:end);
Theta2_tmp2 = lambda/m * Theta2;
Theta2_tmp2 = Theta2_tmp2(:, 2:end);

Theta1_tmp1 = 1/m * (E2' * A1);
Theta1_tmp1 = Theta1_tmp1(:, 2:end);
Theta1_tmp2 = lambda/m * Theta1;
Theta1_tmp2 = Theta1_tmp2(:, 2:end);

Theta2_tmp = Theta2_tmp1 + Theta2_tmp2;
Theta1_tmp = Theta1_tmp1 + Theta1_tmp2;

Theta2_grad = [ Theta2_grad(:,1), Theta2_tmp ];
Theta1_grad = [ Theta1_grad(:,1), Theta1_tmp ];

% ------------------------ Part 3 ------------------------
% --------------------------------------------------------
[a, b] = size(Theta1); % [a, b] = [25, 401]
[c, d] = size(Theta2); % [c, d] = [10, 26]
n1 = a*b;
n2 = c*d;
temp2 = 0;
for t = a+1 : n1    % a+1 as we do not count the bias unit.
    temp2 += Theta1(t)**2;
end

for t = c+1 : n2
    temp2 += Theta2(t)**2;
end

J = (1/m)*temp1 + (lambda/(2*m))*temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
