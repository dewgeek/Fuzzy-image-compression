function results = classify(data, net,  debug)
% CLASSIFY Classifies the given data using the given trained SFAM.
% RESULTS = CLASSIFY(DATA, NET, LABELS, DEBUG) 
%	DATA is an M-by-D matrix where M is the number of samples and D is the size of the feature
%	space. NET is a previously trained SFAM network. LABELS is a M-vector containing the correct
%	labels for the data. If you don't have them, give it as an empty-vector []. 
%	DEBUG is a scalar to control the verbosity of the program during training. If 0, nothing will
%	be printed, otherwise every DEBUG iterations an informatory line will be printed. 
%
% Emre Akbas, May 2006
%


hits=0;
results = zeros(1,size(data,1));
tic;
output = zeros(size(data));

for s=1:size(data,1)
    
    input = zeros(1, 2*size(data,2));
    input(1:size(data,2)) = data(s,:);
    % Complement code input
    input(size(data,2)+1:2*size(data,2)) = 1 - input(1:size(data,2));

    % Compute the activation values for each prototype.
    activation = ones(1,length(net.weights));
    for i=1:length(net.weights)
	activation(i)  = sum(min(input,net.weights{i}))/...
		    (net.alpha + sum(net.weights{i}));
    end

    % Sort activation values 
    [~, sortedIndices] = sort(activation,'descend');
    ro_matrix = zeros(1,length(sortedIndices));
    % Iterate over the prototypes with decreasing activation-value
    results(s) = -1;
    ro  = net.vigilance;
    for p=sortedIndices
        % Compute match of the current candidate prototype 
        match = sum(min(input,net.weights{p}))/sum(input);
        ro_matrix(p) = match;
        % Check resonance
        if match>=ro
            results(s) = net.labels(p);

            temp=min(input,net.weights{p});
            output(s,:) = temp(1:size(data,2));
            break;
%         else
%             ro = sum(min(input,net.weights{p}))/net.D + net.epsilon;
        end
        
    end
    if results(s) == -1
       
        [~, ro_indices] = sort(ro_matrix,'descend');
        results(s)= ro_indices(1);
    end
    if mod(s,debug)==0
	elapsed = toc;
	fprintf(1,'Tested %4dth sample. Hits so far: %3d which is %.3f%%.\tElapsed %.2f seconds.\n',s,hits,100*hits/s,elapsed);
	tic;
    end
end % samples loop

