addpath(genpath("/home/wyatt/workstation"));
%%

i = 1;
x = readcfl(sprintf("/home/wyatt/test_data/kspace/cd%i",i));
y = readcfl(sprintf("/home/wyatt/test_data/kspace/c%i",i));
p = readcfl(sprintf("/home/wyatt/test_data/kspace/po%i",i));




% account for roundoff error
x = real(x);
x = x .* conj(p);


x = squeeze(x(1,:,:));
y = squeeze(y(1,:,:));

ky = fftnc(y);
ky(abs(ky) < 1e-3) = 0;

kx = fftnc(x);

kx(ky == 0) = 0;

x = ifftnc(kx);


niftiwrite(abs(x),"x");
niftiwrite(abs(y),"y");

disp_vol_center(x,false,1);
disp_vol_center(y,false,2);

disp_vol_center(kx,false,3)
disp_vol_center(ky,false,4)