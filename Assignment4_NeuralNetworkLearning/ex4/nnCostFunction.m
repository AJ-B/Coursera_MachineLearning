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
K = num_labels;
         
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

%some added definitions
Delta1 = 0;
Delta2 = 0;

%add column of ones to X matrix
X = [ones(m,1) X];

for i = 1:m,
	
	yk = zeros(K,1);
	yk( y(i) ) = 1;

	a1 = X(i,:)'; %'

	z2 = Theta1*a1;
	a2 = sigmoid( z2 );
	a2 = [1; a2];

	z3 = Theta2*a2;
	a3 = sigmoid( z3 );
	H = a3;

	J += -1/m * ( yk'*log(H) + (1-yk)'*log(1-H) );


	del3 = a3-yk;
	Delta2 += del3*a2';

	del2 = Theta2(:, 2:end)'*del3 .* sigmoidGradient(z2);
	Delta1 += del2*a1';
endfor

%'

treg1 = [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
D1 = 1/m*Delta1 + lambda/m*treg1;

treg2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
D2 = 1/m * Delta2 + lambda/m*treg2;

%fprintf("J=",J);

J += lambda/(2*m) * ( sum(treg1(:).^2) + sum(treg2(:).^2) );
Theta1_grad = D1; 
Theta2_grad = D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
