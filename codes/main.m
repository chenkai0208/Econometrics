% This is the main function for the realization of the LS-SVM model
% together with some simple performance analysis.
% Kai Chen; final edit: 4/30/2018
% This application needs the library: http://www.esat.kuleuven.be/sista/lssvmlab

load sp500_10yr.mat;
dn = datenum(Date);
tb = table(dn,AdjClose) ;
N = size(tb,1);

% log-return on AdjClose
logret = zeros(N,1);
logret(1) = 0;
for i=2:N
    logret(i) = log(AdjClose(i)/AdjClose(i-1));
end

tb = table(dn,AdjClose,logret); % add this to tb

% cross-validation partition
cvp = cvpartition(N,'Holdout',0.1);
idxTrn = training(cvp); % training set indices
idxTest = test(cvp); % test set indices


%Mdl = fitrsvm(tb(idxTrn,:),'AdjClose','KernelFunction','rbf'); % svm model

X = tb(idxTrn,:).dn;
%X = [tb(idxTrn,:).dn,tb(idxTrn,:).logret]; % change the input
Y = tb(idxTrn,:).AdjClose;
type = 'function estimation';
[Yp,alpha,b,gam,sig2,Mdl] = lssvm(X,Y,type);

%YFit = predict(Mdl,tb(idxTest,:)); % svm prediction

YFit = zeros(sum(idxTest),1);
for i=1:242
    YFit(i) = predict(Mdl, tb(idxTest,:).dn(i),1);
    %YFit(i) = predict(Mdl, [tb(idxTest,:).dn(i);tb(idxTest,:).logret(i)],1);
end


plot(tb.dn(idxTest),AdjClose(idxTest));                                                   
hold on                                                
plot(tb.dn(idxTest),YFit,'r')

% have a look at the full picture
hold on
plot(tb.dn,AdjClose,'g');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

table(tb.AdjClose(idxTest),YFit,'VariableNames',{'ObservedValue','PredictedValue'})
YRef = tb.AdjClose(idxTest);
lssvm_err = YFit.^2 - YRef.^2;

% calculate the return series
len = length(YFit);
test_dn = tb.dn(idxTest);
retFit = zeros(len,1);
retRef = zeros(len,1);
for i=1:len
    index = find(tb.dn(:,1)==test_dn(i));
    true = tb(index-1,2).AdjClose;
    retFit(i) = log(YFit(i) / true);
    retRef(i) = log(YRef(i) / true);
end

% MSE
YRef = tb.AdjClose(idxTest);
lssvm_nmse = goodnessOfFit(YFit,YRef,'NMSE'); % normalized mean square error
lssvm_mse = goodnessOfFit(YFit,YRef,'MSE');   % mean square error

% Percentage of Correct Sign Predictions (PCSP)
n = 0; % the number of correct sign prediction

for i=1:len
    true = tb.AdjClose(dn == test_dn(i)-1);
    signFit = sign(YFit(i)-true);
    signRef = sign(YRef(i)-true);
    if signFit == signRef
        n = n+1;
    end
end

lssvm_PCSP = n / len;

% Pesaran-Timmermann Test
lssvm_PT = PT_test(retFit, retRef);

