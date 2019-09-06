function [U,SV,svals]=splitSVD_2p(frames,varargin)
%Modified from splitSVD for analyzing 2p data
%This function is to conduct SVD cleaning to video frames
%Based on 
% Spontaneous behaviors drive multidimensional, brain-wide neural activity
% Carsen Stringer*1;2, Marius Pachitariu*1;3;4, Nicholas Steinmetz5, Charu Reddy5, Matteo Carandiniy5 and
% Kenneth D. Harrisy3;4
%
%a session is split into multiple blocks and SVD are performed for each
%block.
%Concatenate all spatial factors U from each block
%Apply SVD into concatenated U to obtain the estimate of U to original
%frame
%S*V' is given by the projection of frames into U
%Input
%frames:pixels x concatenated frames
%(reshape to 2d if frames are 256x256xframes
%Outpu
%U:pixel x num_sval
%V:temporal factors weighted by svals
%frames x num_sval (S*V')' of svd)
%2018 Hirofumi Nakayama

%Need to change in case specifying ROI
mask=true(512);
fps=30;
% fps=trial_info.fps;


if iscell(frames)
%     num_trials=numel(frames);

    if ndims(frames{1})==3
        if size(frames{1},1)==size(frames{1},2)
        %each component in a cell is 256x256xframes
        frames=cell2mat(cellfun(@(x) reshape(x,[],size(x,3)),frames,'UniformOutput',0)');
        else
           error(sprintf('check frames, ndims=%d,size(frames{1})=%d,%d,%d',ndims(frames{1}),size(frames{1},1),size(frames{1},2),size(frames{1},3)))
        end
    elseif ndims(frames{1})==2
        %each component in a cell is pixels x frames
        frames=cat(2,frames{:});
    end
elseif ndims(frames)==3
    %This will fail if a subset of trials in a session is given
%     num_trials=length(trial_info.inh_onset);%This may need to be written in a different way

    if size(frames,1)==size(frames,2)
        frames=reshape(frames,[],size(frames,3));
    else
       error(sprintf('check frames, ndims=%d,size(frames{1})=%d,%d,%d',ndims(frames{1}),size(frames{1},1),size(frames{1},2),size(frames{1},3))) 
    end
end


% block_size=30;%for 256x256x300 / trial

%block_size=25;
num_block=1;
fig_plot=0;
num_svals=100;
if numel(varargin)==1
    num_svals=varargin{1};
elseif numel(varargin)==2
   [num_svals,num_block]=varargin{1};%Butterworth filter to V 
elseif numel(varargin)==3
   [num_svals,num_block,fig_plot]=varargin{:};
end

frames_sub=floor(linspace(1,size(frames,2)+1,num_block+1));

for i=1:num_block
    tic
    
    subF=single(frames(:,frames_sub(i):frames_sub(i+1)-1));
    subF=subF(:,(max(subF)-min(subF))~=0);%Remove all 0 frames which correspond to skipped trials
    subF=subF(mask(:),:);
    [Ub,Sb,~] = svd(subF,'econ');
    G{i}=Ub(:,1:num_svals)*Sb(1:num_svals,1:num_svals);
    
    sprintf('svd for block %d /%d = %.2f sec',i,num_block,toc)    
end

clear subF

G_all=cell2mat(G);
[U,svals,~] = svd(G_all,'econ');
SV=single(frames(mask(:),:))'*U;

U=U(:,1:num_svals);
SV=SV(:,1:num_svals);

if fig_plot==1
    U=roimask2full(U,mask);
    figure2;
    subt=@(m,n,p) subtightplot(m,n,p,[0.03,0.03],[0.03,0.03],[0.03,0.03]);
    for i=1:20
   subt(4,5,i)
   g1=U(:,:,i);
   g1(~mask)=mean(g1(mask));
   imshow(imadjust(mat2gray(g1)));title(num2str(i))
    end
end