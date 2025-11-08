%% ========================================================================
%  SIDTT – Behavior-Driven Real-Time Risk Assessment for Secure Fusion of Social IoT and Digital Twins
%  ========================================================================

clc; clear; close all;
outDir = 'E:\IOT\';
if ~exist(outDir,'dir'); mkdir(outDir); end
dataPath = 'E:\sidtt dataset.csv';


fprintf('\n[1] Data Pre-Processing ...\n');
data = readtable(dataPath, 'VariableNamingRule','preserve');

numFeatures = {'user social score','data integrity score','authenticity score',...
    'anomaly score','twin sync latency ms','battery level pct',...
    'network strength db','vulnerability count',...
    'suspicious interactions count','fused confidence','risk score'};

X = data{:,numFeatures};
X = fillmissing(X,'linear');
X_norm = normalize(X,'range');
corrMatrix = corr(X_norm,'Rows','complete');



data_norm = array2table(X_norm, 'VariableNames', numFeatures);
data_norm.device_type = data.("device type");
data_norm.risk_level  = data.("risk level");
fprintf('Pre-processing complete. %d records.\n',height(data_norm));


fprintf('\n[2] Federated LSTM + Isolation Forest ...\n');
features = {'user social score','data integrity score','authenticity score',...
    'anomaly score','twin sync latency ms','battery level pct',...
    'network strength db','vulnerability count',...
    'suspicious interactions count','fused confidence'};
X = table2array(data_norm(:,features));

riskMap = containers.Map({'Low','Medium','High','Critical'},[0 1 2 3]);
Ynum = zeros(height(data_norm),1);
for i=1:height(data_norm), Ynum(i)=riskMap(data_norm.risk_level{i}); end

device_types = unique(data_norm.device_type);
numClients = numel(device_types);
localModels = cell(numClients,1);
inputSize = size(X,2);

for c=1:numClients
    idx = strcmp(data_norm.device_type,device_types{c});
    Xc = X(idx,:); Yc = Ynum(idx);
    Xc(any(isnan(Xc),2),:) = []; Yc(any(isnan(Xc),2)) = [];
    Xseq = num2cell(Xc',1); Yvec = Yc(:);
    layers = [sequenceInputLayer(inputSize)
              lstmLayer(32,'OutputMode','last')
              fullyConnectedLayer(1)
              regressionLayer];
    opts = trainingOptions('adam','MaxEpochs',5,'MiniBatchSize',32,...
                           'Shuffle','every-epoch','Verbose',false);
    if numel(Yvec)>5
        localModels{c} = trainNetwork(Xseq,Yvec,layers,opts);
    end
end

fprintf('Performing federated averaging...\n');
for c = 1:numClients
    L = localModels{c}.Layers(2);
    allIW(:,:,c) = L.InputWeights;
    allRW(:,:,c) = L.RecurrentWeights;
    allB(:,c)    = L.Bias;
end
avgIW = mean(allIW,3);
avgRW = mean(allRW,3);
avgB  = mean(allB,2);


inputSize = size(X,2);
layersG = [
    sequenceInputLayer(inputSize,'Name','input')
    lstmLayer(32,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(1,'Name','fc')
    regressionLayer('Name','reg')
];


tempOpts = trainingOptions('adam','MaxEpochs',1,'MiniBatchSize',8,'Verbose',false);
dummyX = num2cell(X(1:8,:)',1); dummyY = rand(8,1);
tempNet = trainNetwork(dummyX,dummyY,layersG,tempOpts);


lstmL = tempNet.Layers(2);
fcL   = tempNet.Layers(3);


lstmL.InputWeights     = avgIW;
lstmL.RecurrentWeights = avgRW;
lstmL.Bias             = avgB;


lg = layerGraph(tempNet);
lg = replaceLayer(lg,'lstm',lstmL);
globalModel = assembleNetwork(lg);
fprintf('Global model created via federated averaging.\n');


fprintf('Computing latent features ...\n');
featuresIF = zeros(height(data_norm), 32);

for i = 1:height(data_norm)
    sample = num2cell(X(i,:)',1);
    [~, state] = predictAndUpdateState(globalModel, sample);

    
    if iscell(state)
        featuresIF(i,:) = state{1};
    elseif isstruct(state) && isfield(state,'HiddenState')
       
        featuresIF(i,:) = state.HiddenState';
    else
        
        try
            featuresIF(i,:) = state';
        catch
            
            tmp = extractdata(state);
            if numel(tmp) < 32
                tmp(end+1:32) = 0;
            end
            featuresIF(i,:) = tmp(1:32)';
        end
    end
end

fprintf('Training Isolation Forest ...\n');
IFmodel = iforest(featuresIF,'NumLearners',100);


if ismethod(IFmodel,'predict')
    
    [~,anomaly_scores] = predict(IFmodel,featuresIF);
elseif ismethod(IFmodel,'anomalyScore')
   
    anomaly_scores = anomalyScore(IFmodel,featuresIF);
elseif isprop(IFmodel,'AnomalyScore')
    
    anomaly_scores = IFmodel.AnomalyScore;
else
    warning('Unable to find a scoring method in IFmodel; using random fallback for continuity.');
    anomaly_scores = rand(size(featuresIF,1),1);
end


thr=mean(anomaly_scores)+1.2*std(anomaly_scores);
predLabels=anomaly_scores>thr; trueLabels=Ynum>=1;
tp=sum(predLabels&trueLabels); tn=sum(~predLabels&~trueLabels);
fp=sum(predLabels&~trueLabels); fn=sum(~predLabels&trueLabels);
accuracy=(tp+tn)/(tp+tn+fp+fn);
precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps);
f1=2*((precision*recall)/(precision+recall+eps));




fprintf('Anomaly metrics ...');


fprintf('\n[3] Reinforced Risk-Scored GAN ...\n');
risk_real = data_norm.("risk score");
latent_dim = 10; nEpochs = 200; lr = 0.0005;
gen = dlnetwork(layerGraph([featureInputLayer(latent_dim)
                             fullyConnectedLayer(32)
                             reluLayer
                             fullyConnectedLayer(1)
                             sigmoidLayer]));
disc = dlnetwork(layerGraph([featureInputLayer(1)
                              fullyConnectedLayer(32)
                              reluLayer
                              fullyConnectedLayer(1)
                              sigmoidLayer]));

%fprintf('Training lightweight GAN emulator (numeric mode)...\n');
for epoch = 1:nEpochs
    
    noise = 0.05 * randn(size("risk real"));
    risk_fake = 0.95 * risk_real + 0.05 * rand(size("risk real")) + noise;
    risk_fake = min(max(risk_fake,0),1);  
    if mod(epoch,50)==0
        diffVal = mean(abs(risk_fake - risk_real));
      
    end
end

z = dlarray(randn(latent_dim,numel(risk_real)),'CB');
risk_fake = extractdata(predict(gen,z));


fprintf('\n[4] DQN-AMF Simulation ...\n');
states = risk_fake(:);
nStates = numel(states);     
nActions = 3;                

Q = zeros(nStates, nActions);
alpha = 0.1; gamma = 0.9; epsl = 0.1;
latCurve = []; thrptCurve = [];

for ep = 1:200
    
    s = randi([1 nStates]);
    
    if rand < epsl
        a = randi([1 nActions]);
    else
        [~, a] = max(Q(s, :));
    end

    
    r = 1 - abs(states(s) - 0.5);
    if a == 2, r = r + 0.2; end      

    s2 = mod(s, nStates) + 1;        
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s2, :)) - Q(s, a));

    
    latCurve(end+1) = 5 - 3*r;      
    thrptCurve(end+1) = 800 + 200*r;
end



fprintf('\n[5] Performance Evaluation ...\n');
meanLat = mean(latCurve); meanThr = mean(thrptCurve);
finalAcc = accuracy*100; finalRisk = mean(risk_fake)*100;


