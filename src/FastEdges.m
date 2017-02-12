%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
d = pwd;
tic, model=edgesTrain(opts); toc; % will load model if already trained
cd(d)

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=1;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms


%% detect edge and visualize results
% I = imread('peppers.png');

% seq = '../data/bottle_03/seq/';
% out = '../data/bottle_03/edges/';
mkdir(out)
files = dir([seq, '*.jpg'])';

% i=1;
N = numel(files);

%clust = parcluster('local');
%clust.NumWorkers = 32;
%saveProfile(clust)

tic
%parpool(32)
parfor i = 1:N
    fprintf(1,'Frame %d\n', i);
    I = imread([seq files(i).name]);
    E = edgesDetect(I,model);
%     B = bwmorph(E > 0.3, 'skel', inf);
%     E(~B) = 0;
    [pathstr, name, ext] = fileparts(files(i).name);
    imwrite(E, [out name '.png']);
end
toc
