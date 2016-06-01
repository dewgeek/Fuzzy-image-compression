function vc = blkM2vc(M, blkS) 
M = double(M)/255;
[height,width] = size(M);

 

r = blkS(1) ;  %4
c = blkS(2) ;  %4

if (rem(height, r) ~= 0) || (rem(width, c) ~= 0) 
    error('blocks do not fit into matrix') 
end

x = width/c; %32
y = height/r; %32

N   = x*y; %1024
rc  = r*c; %16
vc  = zeros(rc, N); %16x1024

for ii = 0:y - 1 
    vc(:,(1:x)+ii*x) = reshape(M((1:r)+ii*r,:),rc,x);
end