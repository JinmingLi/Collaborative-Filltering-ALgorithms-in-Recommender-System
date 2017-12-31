function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
            
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

X1 = (X*Theta'- Y).*R; 
reg1 = (sum(sum(X.^2)) + sum(sum(Theta.^2)))*lambda/2; 
J = sum(sum((X1).^2))/2 + reg1;

X_grad = X1*Theta + lambda*X; 
Theta_grad = X1'*X + lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end
