%% Testing BiochipClass
close all;
bSkipCreate = 0;
if (~bSkipCreate); clear all; close all; clc; bSkipCreate = false; end
%% Configurations
% Simulation switches
sBioassay  = 'TEST';
sPostFix   = 'TEST';
VideoName  = '';
iSeed      = 123;
sModelType = 'MDP';
bSimulate  = 1;

% bResynth   = 0; % Set to 0 if bResynthMo = 1
% bResynthMo = 1;
% bShortest  = 0; % If 1: Run bResynth=1 first, stop, run bResynthMo=0 and bResynth=0

bResynth   = 1; % Set to 0 if bResynthMo = 1
bResynthMo = 0;
bShortest  = 0; % If 1: Run bResynth=1 first, stop, run bResynthMo=0 and bResynth=0


% dPerc      = 0.2;
% sDegMode   = 'Clustered'; %'Scattered';
% C2Range    = [500; 50];
% TauRange   = [0.5 0.9; 0.1 0.2];

% dPerc      = 0.3;
% sDegMode   = 'Scattered';
% C2Range    = [200; 50];
% TauRange   = [0.5 0.9; 0.1 0.2];

% Use this for NO-Fault mode to avoid bugs
dPerc = 0.1;
sDegMode   = 'Scattered';
C2Range    = [200; 200];
TauRange   = [0.7 0.7; 0.7 0.7];

% Brand New
bUsedMB    = 0;
NRange     = [0 0];

% Used
%bUsedMB    = 1;
%NRange     = [0 400];

% Other settings
bAdaptive  = 1;
bGenModel  = 1;
bDStep     = 1;
bLoadTb    = 1;
bSkipPlots = 1;
bDelay     = 0;
bDebug     = 0;
bRecord    = 0;
bFixedH    = 0;
bDegrade   = 1;
supCount   = 1;
simCount   = 3;
kMax       = 0;
iSynthFreq = 1000;
retryMax   = 0; % set to 1
% Biochip configuration
H          = 30;
W          = 60;
HBits      = 2;
C1Range    = [0; 0];

%% Testing Directories
tmpFolder = ['D:\WS\PRISM\MEDA\Output',sPostFix]; 
if ~exist(tmpFolder, 'dir')
    mkdir(tmpFolder)
end
fcnCreateHashTable(sPostFix);


%% Seed
rng(iSeed);

%% Scheduler
if (~bSkipCreate)
    % Create schedule
    SC = MedaSchedulerClass(sPostFix,H,W,HBits,bResynthMo);
    % Configure model
    SC.mdSetModelType(sModelType);
    % Load Bioassay
    switch sBioassay
        case 'TEST'
            fcnLoadTestBioassay(SC,1,[3 3; 4 4; 5 5; 6 6], [10 10; 20 20; 30 30], 10);
        case 'CEP'
            fcnLoadCepBioassay(SC);
        case 'CP2'
            TB = fcnLoadCP2Bioassay(SC);
        case 'SD'
            TB = fcnLoadSdBioassay(SC);
        case 'SD2'
            TB = fcnLoadSD2Bioassay(SC);
        case 'MM2'
            TB = fcnLoadMM2Bioassay(SC);
        case 'Nuip1'
            TB = fcnLoadNuipBioassay(SC,1);
        case 'Nuip2'
            TB = fcnLoadNuipBioassay(SC,2);
        case 'Nuip4'
            TB = fcnLoadNuipBioassay(SC,4);
        case 'Nuip8'
            TB = fcnLoadNuipBioassay(SC,8);
        case 'CRAT'
            TB = fcnLoadCovidBioassay(SC,'rat');
        case 'CPCR'
            TB = fcnLoadCovidBioassay(SC,'pcr');
        case 'Single'
            fcnLoadSingleMoBioassay(SC);
        case 'Demo'
            x1j1s = [1 1 4 4];
            x1j1g = x1j1s + [2 2 2 2];
            SC.mdAddMo('z1','Mag',{},{x1j1s},{x1j1g});
        otherwise
            fprintf('Unknown bioassay %s -- Terminating!\n',sBioassay);
            return
    end
    % Preprocess all MOs
    SC.mdPreprocessAllMos();
end

%% MEDA Biochip
MB = BiochipClass(H,W,HBits,TauRange,C1Range,C2Range,dPerc,sDegMode,bFixedH);
% Attach scheduler
MB.amdSetScheduler(SC);
% Add scheduler droplets to MB
JobStates = MB.amdInitJobStates(length(SC.MoList),2);


%% Logs 
if (bRecord)
    v = VideoWriter(VideoName);
    v.FrameRate = 30;
    v.Quality = 50;
    open(v);
end


%% Simulation Loop
if(~bSimulate)
    fprintf('No simulation requested, aborting\n');
    return
end

% Show biochip
MB.amdSetOptions(bSkipPlots);
if (~bSkipPlots)
    MB.amdShowBiochip();
end


NewCount = zeros(simCount,supCount); % Counts commulative number of health bits lost
bResynthNow = zeros(simCount,1,supCount);
LogMoStates = nan(simCount,SC.MoCount,supCount);
LogK = nan(simCount,supCount);
LogN = {};
retryCounter = zeros(1,supCount); % max number of iterations to end with kMax
LogT = zeros(simCount,supCount);
%S = zeros(kMax+1,length(SC.MoList));

if bUsedMB
    NVals = randi([0 400],H,W);
else
    NVals = 0;
end

for supIdx = 1:supCount
    %MB.N(:) = 0; % reset counters
    MB.mdResetDegradation(NVals);
    SC.HealthMatrix = MB.mdReadHealthMatrix();
    HmOld = MB.mdReadHealthMatrix();
    % Pass health matrix info
    SC.HealthMatrix = MB.mdReadHealthMatrix();
    % This makes sure synthesis is done only once in case of Shortest Path
    if (bShortest && supIdx>1)
        bResynth = 0;
    end
    % Synthesize all Jobs
    if (bResynthMo==0 && supIdx==1)
        SC.mdProcessAllJobs(bResynth,bGenModel,bDStep,bLoadTb,bShortest);
    end

    for simIdx = 1:simCount
        SC.mdResetState();
        HmNew = MB.mdReadHealthMatrix();
        
        ticA = tic;
        for k = 0:kMax
            if (bDebug)
                clc
                fprintf('%d\t',SC.MoList.state); fprintf('\n');
                fprintf('%d\t',SC.MoList.k); fprintf('\n');
            end
    %         if (bDelay>0)
    %             pause(bDelay);
    %         else
    %             drawnow;
    %         end
            if (~bSkipPlots)
                pause(0.5);
                drawnow
            end
            if(bRecord)
                frame = getframe(MB.FigureHandle);
                writeVideo(v, frame);
            end

            % Get current job states
            JobStates = MB.amdGetJobStates();
            SC.mdSetJobStates(JobStates);
            % Process MO internal states
            SC.HealthMatrix = MB.mdReadHealthMatrix();
            SC.mdProcessStates();
            % Update JobStates
            JobStates = SC.mdGetJobStates();
            MB.amdSetJobStates(JobStates);
            % Get control actions
            Actions = SC.mdGetActions();
            % Apply actions
            MB.mdApplyActions(Actions);
            % Degrade
            if(bDegrade)
                MB.mdDegrade();
            end

            %clc
            %fprintf([repmat('%1d',1,size(MB.N,2)) '\n'],flip(MB.N',2));

            %SC.mdTick();

            if(bRecord)
                frame = getframe(MB.FigureHandle);
                writeVideo(v, frame);
            end

            if (SC.mdIsDone)
                break;
            end

            %S(k+1,:) = S(k+1,:) + ( [SC.MoList.state]==SC.cBusy );

        end
        LogT(simIdx,supIdx) = toc(ticA);
        LogK(simIdx,supIdx) = k;
        LogN{simIdx,supIdx} = MB.N;

        bResynthNow(simIdx,supIdx) = mod(simIdx,iSynthFreq)==0;
        HmNew = MB.mdReadHealthMatrix();
        HmDiff = HmOld-HmNew;
        NewCount(simIdx+1,supIdx) = sum(HmDiff,'all')+NewCount(simIdx,supIdx);
        HmOld = HmNew;

        fprintf('Iteration %2d.%4d done at k = %d took %1.3fs\n',...
            supIdx,simIdx,LogK(simIdx,supIdx),LogT(simIdx,supIdx));

        LogMoStates(simIdx,:,supIdx) = [SC.MoList.state];

        if (k==kMax)
            retryCounter(1,supIdx) = retryCounter(1,supIdx)+1;
        else
            retryCounter(1,supIdx) = 0;
        end
        if (retryCounter(1,supIdx) >= retryMax)
            % Save current state
            break
        end 
        
        if (bResynthNow(simIdx,supIdx) && bAdaptive && (simIdx~=simCount))
            if(~all(SC.HealthMatrix==MB.HealthMatrix,'all'))
                SC.HealthMatrix = MB.mdReadHealthMatrix();
                SC.mdProcessAllJobs(bResynth,bGenModel,bDStep,bLoadTb,bShortest);
                %SC.mdSynthesizeJobStr(1,1,bResynth,bGenModel,bDStep,bLoadTb,bShortest);
            else
                bResynthNow(simIdx,supIdx) = 0;
            end
        end


        %fprintf([repmat('%1d',1,size(MB.N,2)) '\n'],flip(MB.HealthMatrix',2));
    end
end

%%
NewCount = NewCount(2:end,:);

if(bRecord)
    close(v);
end

%% Saving
RngState = rng;
UniqueOutputFN = ['Results/',sBioassay,'_',int2str(floor(now*100000))];
save(UniqueOutputFN);

%close all

return


%% Testing Code
clc
SC.mdTick();
SC.mdProcessStates();
clc
fprintf('%d\t',SC.MoList.state); fprintf('\n');
fprintf('%d\t',SC.MoList.k); fprintf('\n');












