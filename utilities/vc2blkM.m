
function M = vc2blkM(vc, r, height,width) 
%vc2blkM Reshaping a matrix vc of rc by 1 vectors into a block-matrix M of rM by cM size 
% Each rc-element column of vc is converted into a r by c  block of a matrix M and placed as a block-row element 

[featureVectorsSize ,featureVectorCount] = size(vc) ; 
pxls = featureVectorsSize*featureVectorCount ; 

if ( (rem(pxls, height) ~= 0) || (rem(height, r) ~= 0) ) 
    error('incorrect number of rows of the matrix') 
end

if ( (rem(pxls, width) ~= 0) || (rem(width, r) ~= 0) ) 
    error('incorrect number of rows of the matrix') 
end


cM = pxls/height; 

if ( (rem(featureVectorsSize, r) ~= 0) || (rem(featureVectorCount*r, height) ~= 0) ) 
error('incorrect block size') 
end 
c = featureVectorsSize/r ; 
xM = zeros(r, featureVectorCount*c); 
xM(:) = vc ; 
nrb = height/r ; 
M = zeros(height, cM); 
for ii = 0:nrb-1 
M((1:r)+ii*r, :) = xM(:, (1:cM)+ii*cM) ; 
end 
M=M*255; 
M= uint8(M);
