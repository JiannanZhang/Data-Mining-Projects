format compact
% read the data in
% here I have file Data,Datamodify,Data1,Data2,Data3. I explain them in readme. each
% time you can comment others and just read one data file
%oriData = dlmread('/Users/Jeffrey/Desktop/DataMiningProject/Data.txt');
oriData = dlmread('/Users/Jeffrey/Desktop/DataMiningProject/Datamodify.txt');
%oriData = dlmread('/Users/Jeffrey/Desktop/DataMiningProject/Data2.txt');
%oriData = dlmread('/Users/Jeffrey/Desktop/DataMiningProject/Data3.txt');
% normlize  each column of data
data = oriData / diag(sqrt(diag(oriData'*oriData))); 
Y = data(:,1);
X = data(:,2:end);
foldsize = 39;
num_folds = 10;
% create an empty list to store all the error after all the process (for each lambda)
lam_err = [];
start = 0;
stop = 10;
step = 1;
for lambda = start : step : stop
    %lambda = log(lambda);
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
        % get the testing data 'xtest' which is a matrix including all
        % features for the specific test fold
        matrix_ones = ones(39,1);
        xtest = [ matrix_ones ytest(:,2:end)];
        % find lambda times w square(regularizaiton)
        %lamW = lambda*w'*w;
        % store all the difference between predict value and actural value
        % into a 39 * 1 matrix
        diff_matrix = ytest(:,1) - xtest*w;
        % square each value
        sqr_diff = diff_matrix.^2;
        % add the lambda*w'*w to the sqr_diff
        error = sqr_diff;
        % find the mean error for such a fold
        mean_err = mean(error);
        %diff = (ytest(:,1) - xtest*w)'*(ytest(:,1) - xtest*w) + lambda*w'*w
        err_list = [err_list,mean_err];
    end 
     % append each error for each lambda into this list
     lam_err = [lam_err,mean(err_list)];
     
end 
% print the result
lambda_err = lam_err
% use loops to create the lambda array 
lambda = zeros(1,((stop-start) / step) + 1);
for i = 1:(stop-start) / step + 1 
    lambda(i) = start + (i - 1) * step;
end 
figure; 
lambda;
% plot the error against lambda so that we look at it easier
scatter(lambda,lambda_err)


    




    





