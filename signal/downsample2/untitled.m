
image = imread('C:\Users\bdbir\Desktop\signal\downsample\sprite.png');
image = rgb2gray(image); % Convert to grayscale if the image is RGB

img = imresize(image, 0.5);

fft_img = fft2(double(img));

fshift = fftshift(fft_img);

[row, col] = size(img);
crow = int32(row/2);
ccol = int32(col/2);
fshift(crow-30:crow+30, ccol-30:ccol+30) = 0;

f_ishift = ifftshift(fshift);
img_back = ifft2(f_ishift);
img_back = abs(img_back);

combined = zeros(row*2, col*2, 'like', img);

combined(row/2+1:row+row/2, col/2+1:col+col/2) = imresize(img, [row, col]);

imwrite(uint8(combined), 'C:\Users\bdbir\Desktop\signal\downsample\downsample_image.jpg');
