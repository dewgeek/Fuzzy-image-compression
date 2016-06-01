function net = train(data, net, debug)
% TRAIN Trains the given SFAM network on the given labeled data. 
%	TNET = TRAIN(DATA, LABELS, NET, DEBUG) traines the given SFAM network NET on DATA with
%	labels LABELS. DATA is M-by-D matrix where M is the number of samples and D is the size of
%	the feature space. LABELS is a M-vector containing the labels for each data. NET can be an
%	untrained or previously trained SFAM network.
%	DEBUG is a scalar to control the verbosity of the program during training. If 0, nothing will
%	be printed, otherwise every DEBUG iterations an informatory line will be printed. 
%	
%	TNET is the trained NET.
%
% Emre Akbas, May 2006
%

%dbstop in train at 18
current_label = 1;
for e=1:net.epochs
    network_changed = false;

    tic;
    for s=1:size(data,1)

        if mod(s,debug)==0
            elapsed = toc;
            fprintf(1,'Training on %dth sample, in %dth epoch.\5# of prototypes=%4d\tElapsed seconds: %f\n',s,e,length(net.weights),elapsed);
            tic;
        end
        input = zeros(1, 2*size(data,2));
        input(1:size(data,2)) = data(s,:);
        
        
        % Complement code input
        input(size(data,2)+1:size(data,2)*2) = 1 - input(1:size(data,2));
        
        % Set vigilance
        ro = net.vigilance;

        % By default, create_new_prototype=true. Only if 'I' resonates with
        % one of the existing prototypes, a new prot. will not be created
        create_new_prototype = true;

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
        for p=sortedIndices
            % Compute match of the current candidate prototype 
            match = sum(min(input,net.weights{p}))/net.D;	% see note [1]
            ro_matrix(p) = match;
            % Check resonance
            if match>=ro
                net.weights{p} = net.beta*(min(input,net.weights{p})) + ...
                    (1-net.beta)*net.weights{p};
                network_changed = true;
                create_new_prototype = false;
                break;
%             else
%                 ro = sum(min(input,net.weights{p}))/net.D + net.epsilon;
            end
    %             % Check labels
    %             if input_label==net.labels(p)
    %                 
    %                 % update the prototype
    %                 
    %             else
    %                 % Match-tracking begins. Increase vigilance
    %                 ro = sum(min(input,net.weights{p}))/net.D + net.epsilon;
    %             end
        end
    

    
        if size(sortedIndices,2)<net.max_categories && create_new_prototype 

            new_index = length(net.weights)+1;
            if net.singlePrecision
                net.weights{new_index} = ones(1,2*net.D,'single');
            else
                net.weights{new_index} = ones(1,2*net.D);
            end

            net.weights{new_index} = net.beta*(min(input,net.weights{new_index})) + ...
                    (1-net.beta)*net.weights{new_index};

            net.labels(new_index)  = current_label;
            current_label = current_label  + 1;
            network_changed = true;
        end

        if size(sortedIndices,2)==net.max_categories && create_new_prototype 
            [~, ro_indices] = sort(ro_matrix,'descend');
            new_index = ro_indices(1);

            net.weights{new_index} = net.beta*(min(input,net.weights{new_index})) + ...
                    (1-net.beta)*net.weights{new_index};

            network_changed = true;
        end


    end % samples loop

    if ~network_changed
        fprintf(1,'Network trained in %d epochs.\n',e);
        break
    end
end % epochs loop


%
% NOTES:
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% [1] 'net.D' is written insyead of |input|. These values are equal.
%
