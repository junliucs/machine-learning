% J is the cost and grad is the gradient
function [J, grad] = costFunction(theta, X, y)

J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h));
grad = (1/m)*X'*(h - y);

end
