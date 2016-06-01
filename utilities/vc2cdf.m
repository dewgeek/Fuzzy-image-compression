function [ output_args ] = vc2cdf( vc, ncdf )

output_args = zeros(size(vc));
[no_of_features,no_of_vectors] = size(vc);

for i=1:no_of_vectors
    temp = vc(:,i);
    temp_output = zeros(1,no_of_features);
    for j=1:no_of_features
        
        temp_output(j) = ncdf(temp(j)+1);
    end
    output_args(:,i) = temp_output';
end
max_value = max(max(output_args));
min_value = min(min(output_args));
output_args = (output_args - min_value)/(max_value - min_value);
end

