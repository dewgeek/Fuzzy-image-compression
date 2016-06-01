function output = fuzzy_comp(tnet,imagePath,fileName)
I = imread(imagePath);

wf = 3; %Wiener filter size
% I1 = I(:,:,1);
% I2 = I(:,:,2);
% I3 = I(:,:,3);
I1 = wiener2(I(:,:,1),[wf wf]);
I2 = wiener2(I(:,:,2),[wf wf]);
I3 = wiener2(I(:,:,3),[wf wf]);

r=4;
I1_resized=blkM2vc(I1,[r r]); 
I2_resized=blkM2vc(I2,[r r]); 
I3_resized=blkM2vc(I3,[r r]); 

% ncdf = imgnormcdf(rgb2gray(I));
% I1_resized = vc2cdf(I1_resized*255,ncdf);
% I2_resized = vc2cdf(I2_resized*255,ncdf);
% I3_resized = vc2cdf(I3_resized*255,ncdf);

% test the network on the testdata
wf2 = 4;
I1_classified = classify(I1_resized', tnet, 1000);
l = lookup(I1_classified ,tnet.weights);
I1_compressed = wiener2(vc2blkM(l',r,size(I1,1),size(I1,2)),[wf2 wf2]);

I2_classified = classify(I2_resized', tnet, 1000);
l = lookup(I2_classified ,tnet.weights);
I2_compressed = wiener2(vc2blkM(l',r,size(I2,1),size(I2,2)),[wf2 wf2]);
% 
I3_classified = classify(I3_resized', tnet, 1000);
l = lookup(I3_classified ,tnet.weights);
I3_compressed = wiener2(vc2blkM(l',r,size(I3,1),size(I3,2)),[wf2 wf2]);

% I_output = [I1_classified I2_classified I3_classified];
% [B1, N1] = RunLength_M(I_output);
% fileID = fopen(fileName,'w');
% fprintf(fileID,'%1d,',B1);
% fprintf(fileID,'%1d,',N1);
% fclose(fileID);
output = cat(3,I1_compressed,I2_compressed,I3_compressed);
% output = I1_compressed;