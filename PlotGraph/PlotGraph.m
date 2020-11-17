%% Brain plots
%% Atlas + Selection

% Set these variables
TASK = 'MOTOR'
SUBJECT = 90
CONDITION = 1



subcortical = 1;
n_ROI = 379;
CM=zeros(n_ROI,n_ROI);
%CodeBookpath=which(strcat(parc{parcellation},'_subc_codebook.mat'));

%ALEC - getting location of this filepath
[filepath,name,ext] = fileparts(mfilename('fullpath'));
addpath(fullfile(filepath,'BrainGraphTools'));

%ALEC - name of Codebook to use for plotting
CodeBookpath= fullfile(filepath,'BrainGraphTools','dbs80symm_2mm_codebook.mat');
CodeBook=load(CodeBookpath);
CodeBook=CodeBook.codeBook;

clear CodeBook2
if subcortical==0
    for i=1:size(CodeBook.id,1)-19
        CodeBook2.id(i,1)=CodeBook.id(i);
        CodeBook2.name{i,1}=CodeBook.name{i};
        CodeBook2.sname{i,1}=CodeBook.sname{i};
        CodeBook2.rname{i,1}=CodeBook.rname{i};
        CodeBook2.center{i}=CodeBook.center{i};
        CodeBook2.voxels(i)=CodeBook.voxels(i);    
    end
    CodeBook2.num=size(CodeBook.id,1)-19;
    CodeBook=CodeBook2;
end

%% adjust Cvalues for saturation (to eliminate outliers peaks)
data_path = fullfile(filepath, '..', 'GLM/betas', ['betas_' TASK '.mat']);
data=load(data_path);
CC2 = data.beta(SUBJECT, :, CONDITION)';

saturate = true;

if saturate
thr=1;
CC2new=CC2;
CC2new(find(CC2>thr))=0;
CC2new(find(CC2>thr))=max(CC2new);  
CC2new(find(CC2<-thr))=0;
CC2new(find(CC2<-thr))=min(CC2new);
CC2=CC2new;
end

%% plot with normal color scheme 
CC=abs(CC2)*10;
T_conn=0.9;
Factor_SphereSize=max(CC);
Factor_Col=1;
Exp_Sphere=2;
View=1;

Colormap_edges='jet';
Colormap_nodes='jet';

Gamma=0.5;
LinearWeight=0.3;
CA=[-1 1]; 


%%
PlotBrainGraph(CM,CC,CC2,CodeBook,T_conn,Factor_SphereSize,...
    Factor_Col,Exp_Sphere,View,Colormap_nodes,Colormap_edges,Gamma,...
    LinearWeight,CA)

