load('result.mat');
g_16=result.global_16;%16 bits of global feature
g_24=result.global_24;%24 bits of global feature
g_32=result.global_32;%32 bits of global feature
local_fea=result.local;%local feature
local_all=result.local_all;
local_all=NormalizeFea(local_all);

dim_rdc=32;
% get the responses of local feature
% options_w=[];
% options_w.NeighborMode = 'Supervised';
% options_w.k = 5;
% options_w.bLDA=1;
% options_w.gnd = gnd_Train;
%  W = constructW(local_all,options_w);
 W = constructW(local_all);
 
 Y = Eigenmap(W,dim_rdc);
 %  set the options of SR algorithm
 options_sr=[];
 
 options_sr.ReguAlpha = 0.01;
 options_sr.ReguType = 'Ridge';
 options_sr.W = W;
 [eigvector] = SR(options_sr,Y,local_all);