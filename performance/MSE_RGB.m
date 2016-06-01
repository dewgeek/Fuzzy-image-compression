function output = MSE_RGB(x,y)
I1 = double(imread(x))/255;
I2 = double(imread(y))/255;
output = sum( (I1(:) - I2(:)).^2 );
