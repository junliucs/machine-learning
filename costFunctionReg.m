% J is the cost and grad is the gradient
function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

end
