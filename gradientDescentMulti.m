function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); 
n = size(X , 2);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	H = X * theta;
	T = zeros(n , 1);
	for i = 1 : m,
		T = T + (H(i) - y(i)) * X(i,:)';	
	end
	theta = theta - (alpha * T) / m;
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
