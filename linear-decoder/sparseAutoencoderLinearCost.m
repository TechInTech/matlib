function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------

% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
%
%H=zeros(size(data));
%m=size(data,2);
%sparsity_vec=zeros(hiddenSize,1);
%for index = 1:(m/1000)
  % z2=W1*data(:,index)+b1;
   %a2=sigmoid(z2);

  % for q = 1:hiddenSize
   %    sparsity_vec(q)=(1/m) *sum(sum( a2(q).*data));
   %end
 %  sparsity_delta=beta* (-(sparsityParam./sparsity_vec) + ( (1-sparsityParam)./(1.-sparsity_vec) ));

  % z3=W2*a2+b2;
   %a3=sigmoid(z3);
   %H(:,index)=a3;
   %delta3=-(data(:,index)-a3)   .*   ( a3.*(1-a3) );
   %delta2=(W2'*delta3+sparsity_delta) .* (a2.*(1-a2));
   %if u want to use gradient checking,
   %make sure that
   %delta3*a2'=g(theta)
   %W2grad =W2grad + delta3 * a2';
   %b2grad =b2grad + delta3;
   %W1grad = W1grad + delta2 * data(:,index)';
   %b1grad =b1grad + delta2;
%end
%alpha=10;
%W1=W1 - alpha*  ( ((1/m)*W1grad) +lambda*W1 );
%b1=b1 - alpha*((1/m)*b1grad);
%W2=W2 - alpha*  ( ((1/m)*W2grad) +lambda*W2 );
%b2=b2 - alpha*((1/m)*b2grad);
%J=(1/(2*m)) * sum(sum((H-data).^2)) + (lambda/2) * ( sum(sum(W1.^2)) + sum(sum(W2.^2)) );

%sparsity1=0;
%for j = 1:hiddenSize
 %   mid=(1/m) *sum(sum( a2(j)*data));
 %   sparsity1=sparsityParam*log(sparsityParam/mid) + (1-sparsityParam) * log((1-sparsityParam)/(1-mid));
    %cost=J + beta* sparsity1;
%end
Jcost = 0;%直接误差
Jweight = 0;%权值惩罚
Jsparse = 0;%稀疏性惩罚
[n m] = size(data);%m为样本的个数，n为样本的特征数

%前向算法计算各神经网络节点的线性组合值和active值
z2 = W1*data+repmat(b1,1,m);%注意这里一定要将b1向量复制扩展成m列的矩阵
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = z3;

% 计算预测产生的误差
Jcost = (0.5/m)*sum(sum((a3-data).^2));

%计算权值惩罚项
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));

%计算稀释性规则项
rho = (1/m).*sum(a2,2);%求出第一个隐含层的平均值向量
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

%损失函数的总表达式
cost = Jcost+lambda*Jweight+beta*Jsparse;

%反向算法求出每个节点的误差值
d3 = -(data-a3);
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));%因为加入了稀疏规则项，所以
                                                             %计算偏导时需要引入该项
d2 = (W2'*d3+repmat(sterm,1,m)).*sigmoidInv(z2);

%计算W1grad
W1grad = W1grad+d2*data';
W1grad = (1/m)*W1grad+lambda*W1;

%计算W2grad
W2grad = W2grad+d3*a2';
W2grad = (1/m).*W2grad+lambda*W2;

%计算b1grad
b1grad = b1grad+sum(d2,2);
b1grad = (1/m)*b1grad;%注意b的偏导是一个向量，所以这里应该把每一行的值累加起来

%计算b2grad
b2grad = b2grad+sum(d3,2);
b2grad = (1/m)*b2grad;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end

function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end
