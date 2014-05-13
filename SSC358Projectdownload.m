clear
format compact
oriData = dlmread('/Users/Jeffrey/Desktop/DataMiningProject/Data2.txt');
% norm col
data = oriData / diag(sqrt(diag(oriData'*oriData))); 
Y = data(:,1);
X = data(:,2:end);
%randidx = randperm(length(YY));
foldsize = 39;
num_folds = 10;
% create an empty list to store all the error after all the process (for each lambda)
lam_err = [];
start = 0
stop = 10
step = 1
for lambda = start : step : stop
    % cross validation
    err_list = [];
    for k = 1:num_folds
        datacopy = data;
        ytest = datacopy((k-1)*39+1:k*39,:);
        datacopy((k-1)*39+1:k*39,:) = [];
        % train data using the ten-fold crossing validation
        xtrain = datacopy;
        yi = xtrain(:,1);
        xi = xtrain(:,2:end);
        w = ridge(yi,xi,lambda,0);
        w0 = w(1);
        %w_remain = w(2:end)
        [m,n] = size (ytest);
        % get the testing data 'xtest' which is a matrix including all
        % features for the specific test fold
        matrix_ones = ones(39,1);
        xtest = [ matrix_ones ytest(:,2:end)];
        % find lambda times w square(regularizaiton)
        lamW = lambda*w'*w;
        % store all the difference between predict value and actural value
        % into a 39 * 1 matrix
        diff_matrix = ytest(:,1) - xtest*w;
        % square each value
        sqr_diff = diff_matrix.^2;
        % add the lambda*w'*w to the sqr_diff
        error = sqr_diff + lamW;
        % find the mean error for such a fold
        mean_err = mean(error);
        %diff = (ytest(:,1) - xtest*w)'*(ytest(:,1) - xtest*w) + lambda*w'*w
        err_list = [err_list,mean_err];
    end 
     % append each error for each lambda into this list
     lam_err = [lam_err,mean(err_list)];
     
end 
% print the result
lambda_err = lam_err;
% use loops to create the lambda array 
lambda = zeros(1,((stop-start) / step) + 1)
for i = 1:(stop-start) / step + 1 
    lambda(i) = start + (i - 1) * step;
end 
figure; 
lambda;
scatter(lambda,lambda_err);

        
        
        
        
        
        
        
%         testidx =  randidx(k*39 - (foldsize - 1):k*39);
%         y = YY(1:k*39,:)
%         Y = YY(foldidx,:)
%         X = XX(foldidx, :)
% 
%     end
%        Y = YY(trainidx)
%        X = XX(trainidx)
%        ytest = YY(testidx)
%        xtest = XX9testidx)
% 
%     %    Y = data(foldidx)
%     %     
%     % fold1idx = randidx(1:39)
%     % fold2idx = randidx(40:79)
%     % fold3idx = randidx(80:119)
% 
% 
% for (lambda = 1)
%     %we wanna minimize diff
%     diff = []
%     % w is the weights for all the features and w0
%     w = ridge(Y,X,lambda,0);
%     w0 = h(1)
%     % w' is the weights for all the features 
%     w' = w(2:end)
%     [m,n] = size (X);
%     diff = (y-)
%     diff = [diff,]
% 
%     for (index = 1:m) 
%         z = X(index,:);
%         % pre for each instance
%         predictvalue(index) = (z * w) + w0;
%         c(index) = lambda;        
%     end
%     [c' predictvalue' Y  predictvalue'-Y];
% end
    




    





