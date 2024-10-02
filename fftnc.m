function x = fftnc(x)
x = fftshift(fftn(ifftshift(x)));