%
% SFAM usage demo. 
%
function sfam_demo
clear all;
% load data
load demodata

% create network
net = create_network(size(data,2));

% change some parameters as you wish
 net.epochs = 4;

% train the network
tnet = train(data,  net, 100);

% test the network on the testdata
r = classify(testdata, tnet,  100);

% compute classification performance
%fprintf(1,'Hit rate: %f\n', sum(r' == testlabels)*100/size(testdata,1));
