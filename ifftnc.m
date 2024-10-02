function x = ifftnc(x)
x = ifftshift(ifftn(fftshift(x)));