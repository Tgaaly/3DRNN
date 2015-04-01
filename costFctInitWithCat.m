function [cost,grad,catRightBot,catTotalBot,catRightTop,catTotalTop] = ...
    costFctInitWithCat(X,decodeInfo,goodPairsL,goodPairsR,badPairsL,badPairsR,...
    onlyGoodL,onlyGoodR,allSegs,params)

%compute error at last layer (softmax out) --> classifier Wcat (is Wlabel
%in the paper)
cost=0;
[Wbot,W,Wout,Wcat] = stack2param(X, decodeInfo);

%% for good pairs
numOnlyGood = size(goodPairsL,2);
goodBotL= params.f(Wbot* goodPairsL);
goodBotR= params.f(Wbot* goodPairsR);

% onlyGoodHid = params.f(W * [onlyGoodBotL; onlyGoodBotR; ones(1,numOnlyGood)]);
goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numOnlyGood)]);

catHid = Wcat * [goodHid ; ones(1,numOnlyGood)];
catOut_good = softmax(catHid);


%% for bad pairs
numOnlyBad = size(badPairsL,2);
badBotL= params.f(Wbot* badPairsL);
badBotR= params.f(Wbot* badPairsR);

% onlyGoodHid = params.f(W * [onlyGoodBotL; onlyGoodBotR; ones(1,numOnlyGood)]);
badHid = params.f(W * [badBotL; badBotR; ones(1,numOnlyBad)]);

catHid = Wcat * [badHid ; ones(1,numOnlyBad)];
catOut_bad = softmax(catHid);


catOut = [catOut_good catOut_bad];


%% set labels 1-good merge, 2-bad merge
target = zeros(params.numLabels,numOnlyGood+numOnlyBad);
onlyGoodLabels = ones(1,numOnlyGood);
onlyBadLabels = ones(1,numOnlyBad)*2;
target(sub2ind(size(target),onlyGoodLabels,1:numOnlyGood))=1;
target(sub2ind(size(target),onlyBadLabels,numOnlyGood+1:numOnlyGood+numOnlyBad))=1;



cost = cost  -sum(sum(target.*log(catOut)));

[~, classOut] = max(catOut);
catRightTop = sum(classOut==[onlyGoodLabels onlyBadLabels]);
catTotalTop = length(classOut);
deltaCatTop = (catOut-target);

%%% df_Wcat
df_Wcat =  deltaCatTop * [ goodHid' ones(numOnlyGood,1) ; badHid' ones(numOnlyBad,1)];

deltaDownCatTop = Wcat' * deltaCatTop .*params.df([ [goodHid ;ones(1,numOnlyGood)] , [badHid ;ones(1,numOnlyBad)]]);
deltaDownCatTop= deltaDownCatTop(1:params.numHid,:);

%%% df_W
df_W = deltaDownCatTop*[[goodBotL; goodBotR; ones(1,numOnlyGood)] , [badBotL; badBotR; ones(1,numOnlyBad)]]';

% ERROR SPLITS IN TWO HERE FOR LEFT CHILD AND RIGHT CHILD
deltaDownTop = (W'*deltaDownCatTop) .* params.df(...
    [[goodBotL; goodBotR; ones(1,numOnlyGood)] , [badBotL; badBotR; ones(1,numOnlyBad)]]);
deltaDownTopL = deltaDownTop(1:params.numHid,:);
deltaDownTopR = deltaDownTop(params.numHid+1:2*params.numHid,:);

% % now the kids! AS WE ARE GOING DOWN WE NEED TO FURTHER COMPUTE FOR THE
% % CHILDREN NODES!!!
% catHidL = Wcat * [onlyGoodBotL ; ones(1,numOnlyGood)];
% catHidR = Wcat * [onlyGoodBotR ; ones(1,numOnlyGood)];
% % catHidA = Wcat * [onlyGoodBotA ; ones(1,numAll)];
% 
% catOutL = softmax(catHidL);
% catOutR = softmax(catHidR);
% % catOutA = softmax(catHidA);
% 
% % target is the same as for the merged!
% cost = cost -sum(sum(target.*log(catOutL)));
% cost = cost -sum(sum(target.*log(catOutR)));
% % costA = -sum(sum(targetA.*log(catOutA)));
% [~, classOutL] = max(catOutL);
% [~, classOutR] = max(catOutR);
% % [~, classOutA] = max(catOutA);
% catRightBot = 0           +sum(classOutL==onlyGoodLabels);
% catRightBot = catRightBot +sum(classOutR==onlyGoodLabels);
% % catRightBot = catRightBot +sum(classOutA==allSegLabels);
% catTotalBot = length(classOutL)+length(classOutR);%+length(classOutA);
% 
% deltaCatBotL = (catOutL-target);
% deltaCatBotR = (catOutR-target);
% % deltaCatBotA = (catOutA-targetA);
% 
% %%% df_Wcat
% df_Wcat =  df_Wcat + deltaCatBotL * [ onlyGoodBotL' ones(numOnlyGood,1)];
% df_Wcat =  df_Wcat + deltaCatBotR * [ onlyGoodBotR' ones(numOnlyGood,1)];
% % df_WcatA =  deltaCatBotA * [onlyGoodBotA' ones(numAll,1)];
% 
% deltaDownCatL = Wcat' * deltaCatBotL .*params.df([ onlyGoodBotL ;ones(1,numOnlyGood)]);
% deltaDownCatR = Wcat' * deltaCatBotR .*params.df([ onlyGoodBotR ;ones(1,numOnlyGood)]);
% % deltaDownCatA = Wcat' * deltaCatBotA .*params.df([ onlyGoodBotA ;ones(1,numAll)]);
% 
% deltaDownCatL =deltaDownCatL(1:params.numHid,:);
% deltaDownCatR =deltaDownCatR(1:params.numHid,:);
% % deltaDownCatA =deltaDownCatA(1:params.numHid,:);
deltaDownCatL=0;
deltaDownCatR=0;

% FULL ERROR = ERROR DUE TO CATEGORY IN THIS NODE + ERROR COMING FROM ABOVE
% (LEFT AND RIGHT PART SEPARATELY)
deltaFullDownL = deltaDownCatL+deltaDownTopL;
deltaFullDownR = deltaDownCatR+deltaDownTopR;
% these are just single segs
% deltaFullDownA = deltaDownCatA;

%%% df_Wbot (FOR BOTTOM LAYER)
df_Wbot = deltaFullDownL * [goodPairsL , badPairsL]';
df_Wbot = df_Wbot +  deltaFullDownR * [goodPairsR , badPairsR]';%onlyGoodR';

% set them to zero because im only considering good and bad pairs
% df_WbotA = 0;%deltaFullDownA * allSegs';
% df_WcatA = 0;
% costA=0;

%%% final cost and derivatives of categories
cost = 1./(3* (numOnlyGood+numOnlyBad))  * cost;% + 1./numAll * costA;
df_Wcat_CAT = 1./(3 * (numOnlyGood+numOnlyBad))  * df_Wcat;% + 1./numAll * df_WcatA;
df_W_CAT = 1./(3 * (numOnlyGood+numOnlyBad)) * df_W;
df_Wbot_CAT = 1./(3 * (numOnlyGood+numOnlyBad))  * df_Wbot;% + 1./numAll * df_WbotA;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code below starts looking at the goodPairsL/R and badPairsL/R. Code
% above only looked at the good pairs only (maybe for initialization??)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FORWARD PROPAGATION

% forward prop all segment features into the hidden/"semantic" space
goodBotL = params.f(Wbot* goodPairsL);
goodBotR = params.f(Wbot* goodPairsR);
badBotL = params.f(Wbot* badPairsL);
badBotR = params.f(Wbot* badPairsR);

numGoodAll = size(goodBotL,2);
numBadAll = size(badBotL,2);

% forward prop the PAIRS and compute scores (ONLY PAIRS HERE)
goodHid = params.f(W * [goodBotL ; goodBotR ; ones(1,numGoodAll)]);
badHid  = params.f(W * [badBotL ; badBotR ; ones(1,numBadAll)]);

%COMPUTE SCORES!!
% scoresGood = Wout*goodHid;
% scoresBad = Wout*badHid;
% 
% % compute cost
% costAll = 1-scoresGood+scoresBad;
% ignoreGBPairs = costAll<0;
% 
% costAll(ignoreGBPairs)  = [];
% % goodBotL(:,ignoreGBPairs) = [];
% % goodBotR(:,ignoreGBPairs) = [];
% % badBotL(:,ignoreGBPairs) = [];
% % badBotR(:,ignoreGBPairs) = [];
% goodHid(:,ignoreGBPairs) = [];
% badHid(:,ignoreGBPairs)  = [];
% % goodPairsL(:,ignoreGBPairs)  = [];
% % goodPairsR(:,ignoreGBPairs)  = [];
% % badPairsL(:,ignoreGBPairs)  = [];
% % badPairsR(:,ignoreGBPairs)  = [];

% numAll = length(costAll);
% 
% % COMPUTE COST!
% cost = cost + 1./length(ignoreGBPairs) * sum(costAll(:)) + params.regPTC/2 ...
%     * (sum(Wbot(:).^2) +sum(W(:).^2) +sum(Wout(:).^2) +sum(Wcat(:).^2));

% DONT CARE ABOUT SCORE!!!
% OUTPUT LAYER ERROR (DF)
df_Wout = 0;%-sum(goodHid,2)' +  sum(badHid,2)';

% MIDDLE LAYERS W
% subtract good neighbors:
% delta4 = bsxfun(@times,Wout',params.df(goodHid));
% df_W = -delta4 * [goodBotL ; goodBotR ; ones(1,numAll)]';

% delta3 =(W'*delta4) .* params.df([goodBotL ; goodBotR ; ones(1,numAll)]);
% delta3L = delta3(1:params.numHid,:);
% delta3R = delta3(params.numHid+1:2*params.numHid,:);

% DF FOR BOTTOM LAYER
df_Wbot = 0;%- delta3L * goodPairsL';
% df_Wbot = df_Wbot - delta3R * goodPairsR';

% add bad neighbors
% delta4 = bsxfun(@times,Wout',params.df(badHid));
df_W = 0;%df_W +  delta4 * [badBotL ; badBotR ; ones(1,numAll)]';

% delta3 =(W'*delta4) .* params.df([badBotL ; badBotR ; ones(1,numAll)]);
% delta3L = delta3(1:params.numHid,:);
% delta3R = delta3(params.numHid+1:2*params.numHid,:);

% df_Wbot = df_Wbot +  delta3L * badPairsL';
% df_Wbot = df_Wbot +  delta3R * badPairsR';
% 

% add category's derivatives and regularizer
% df_Wcat_CAT COMES FROM USING THE ONLYGOODL/R BACKPROP PART ABOVE (BEFORE
% FEEDFORWARD PART DEALING WITH GOOD AND BAD PAIRS)
df_Wcat = df_Wcat_CAT + params.regPTC * Wcat;

df_Wbot = df_Wbot_CAT;%  + 1./length(ignoreGBPairs) * df_Wbot + params.regPTC * Wbot;
df_W    = df_W_CAT;%  + 1./length(ignoreGBPairs) * df_W    + params.regPTC * W;
df_Wout = zeros(size(Wout));% 1./length(ignoreGBPairs) * df_Wout + params.regPTC * Wout;


[grad,~] = param2stack(df_Wbot,df_W,df_Wout,df_Wcat);