% This is the benchmark model for the project.
% Kai Chen

Mdl2 = arima(14,0,0); % AR(14) as a benchmark
EstMdl2 = estimate(Mdl2, AdjClose(1:sum(idxTrn)));

arima_predict = zeros(sum(idxTest),1);

% single point prediction
for i=1:sum(idxTest)
    arima_predict(i) = forecast(EstMdl2,1,'Y0',AdjClose(1:sum(idxTrn)+i-1));
end

% get the return series
len = length(arima_predict);
retARIMA = zeros(len,1);
retTest = zeros(len,1);
retARIMA(1) = log(arima_predict(1) / AdjClose(sum(idxTrn)));
retTest(1) = log(AdjClose(sum(idxTrn)+1) / AdjClose(sum(idxTrn)));

for i=2:len
    retARIMA(i) = log(arima_predict(i) / arima_predict(i-1));
    retTest(i) = log(AdjClose(sum(idxTrn)+i) / AdjClose(sum(idxTrn)+i-1));
end

arima_err = arima_predict.^2 - AdjClose(sum(idxTrn)+1:end).^2;

plot(arima_predict);
hold on
plot(AdjClose(sum(idxTrn)+1:end));

% MSE
arima_ref = AdjClose(sum(idxTrn)+1:end);
arima_nmse = goodnessOfFit(arima_predict,arima_ref,'NMSE');
arima_mse = goodnessOfFit(arima_predict,arima_ref,'MSE');

% Percentage of Correct Sign Predictions (PCSP)
n = 0;
for i=1:len
    true = AdjClose(sum(idxTrn)+i-1);
    signFit = sign(arima_predict(i)-true);
    signRef = sign(arima_ref(i)-true);
    if signFit == signRef
        n = n+1;
    end
end

arima_PCSP = n / len;

% Pesaran-Timmermann Test
arima_PT = PT_test(retARIMA, retTest);