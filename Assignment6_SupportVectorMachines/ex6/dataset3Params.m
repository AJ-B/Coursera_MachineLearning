function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


vals = [0.01,0.03,0.1,0.3,1,1,10,30];
n = length(vals);
error = zeros( n, n );
for i = 1:n,
	c = vals(i);
	for j = 1:n,
		sig = vals(j);
		%disp("C=");
		%disp(c);
		%disp("Sigma=");
		%disp(sig);
		model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
		predictions = svmPredict(model, Xval);
		error(i,j) = mean(double(predictions ~= yval));
	endfor
endfor

%disp(error)

[i,j] = find(error == min(min(error)));
%disp(i);
%disp(j);
C = vals(i(1));
sigma = vals(j(1));

% =========================================================================

end
