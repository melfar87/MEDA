classdef MedaSchedulerClass < handle & matlab.System
% MEDASCHEDULERCLASS Create BiochipClass object
%   BC = BiochipClass(HEIGHT,WIDTH) creates a biochip object of size
%   HEIGHTxWIDTH.
%
% Author: Mahmoud ELFAR
% email: mahmoud.elfar@duke.edu
% 2020-08-07

    %% Read-only properties
    properties (SetAccess=private, GetAccess=public)
        sPostFix = ''; 
        PrismBinPath = 'D:\PRISM\prism-games-compo\prism\bin';
        OutputPath   = 'D:\WS\PRISM\MEDA\Output';
        ProjectPath  = 'D:\WS\PRISM\MEDA';
        cInit  = 0
        cReady = 1
        cBusy  = 2
        cWait  = 3
        cDone  = 4
        MoTypes = {'Dis','Out','Dsc','Mix','Dlt','Mag','Was','Spt'}
        MoListStruct = struct(...
            'id', {},...
            'name', {},...
            'type', {},...
            'pre', {},...
            'droplets', {},...
            'locations', {},...
            'cond', {},...
            'state', {},...
            'k', {},...
            'time', {},...
            'jobList', {})
        JobListStruct = struct(...
            'id', {},...
            'name', {},...
            'droplet', {},...
            'goal', {},...
            'hazard', {},...
            'kmax', {},...
            'drSize', {},...
            'strTB', {},...
            'pMaxVal', {},...
            'show', {})
        kmax = 30
    end
    
    properties (SetAccess=public, GetAccess=public)
        Height
        Width
        Bits
        HealthMatrix
        Actions
        JobStates
        MoList
        MoCount
        JobCount
        Schedule
        Ticks
        State
        ModelType = 'SMG'
        bResynthMo = 0
        bForceResynth = 1 % Controls whether resynth is forced
    end
    
    %% Constructor
    
    methods
        function obj = MedaSchedulerClass(sPostFix,Height,Width,HBits,bResynthMo)
            obj.Height = Height;
            obj.Width = Width;
            obj.Bits = HBits;
            obj.MoList = obj.MoListStruct;
            obj.Ticks = 0;
            obj.State = obj.cInit;
            obj.MoCount = 0;
            obj.JobCount = 0;
            obj.bResynthMo = bResynthMo;
            obj.sPostFix = sPostFix;
            obj.OutputPath = [obj.OutputPath, obj.sPostFix];
        end
    end
    
    
    %% Public Asynchronous Methods
    methods (Access=public)
        
        %% FUNCTION Reset State
        function mdResetState(obj)
            [obj.MoList.state] = deal(obj.cReady);
            [obj.MoList.k] = deal(-1);
            obj.Actions = [];
            obj.JobStates = [];
            obj.Ticks = 0;
            obj.State = obj.cInit;
            for moId = 1:length(obj.MoList)
                for jobId = 1:length(obj.MoList(moId).jobList)
                    obj.MoList(moId).jobList(jobId).show = 0;
                end
            end
        end
        
        %% FUNCTION Set Model Type
        function mdSetModelType(obj,sModelType)
            switch sModelType
                case {'SMG','MDP'}
                    obj.ModelType = sModelType;
                otherwise
                    fprintf('Error: Unknown model type %s\n',sModelType);
            end
        end
        
        %% FUNCTION Add new MO
        function moId = mdAddMo(obj,vName,vType,vPre,vStartDr,vEndDr,vCond)
            % Get new ID
            moId = size(obj.MoList,2)+1;
            % Set parameters
            obj.MoList(moId).id   = moId;
            obj.MoList(moId).name = vName;
            obj.MoList(moId).type = vType;
            obj.MoList(moId).pre  = vPre;
            obj.MoList(moId).droplets = vStartDr;
            obj.MoList(moId).locations = vEndDr;
            if(exist('vCond','var'))
                obj.MoList(moId).cond = vCond;
            else
                obj.MoList(moId).cond = cell(0,0);
            end
            obj.MoList(moId).state = obj.cInit;
            obj.MoList(moId).k = -1;
            obj.MoList(moId).time = -1;
            obj.MoList(moId).jobList = obj.JobListStruct;
            obj.MoCount = obj.MoCount+1;
        end
        
        %% FUNCTION Increment clock
        function mdTick(obj)
            for moId = 1:length(obj.MoList)
                if ((obj.MoList(moId).k>=0)&&(obj.MoList(moId).state==obj.cBusy))
                    obj.MoList(moId).k = obj.MoList(moId).k + 1;
                end
            end
            obj.Ticks = obj.Ticks + 1;
        end
        
        %% FUNCTION Reset clock
        function mdResetClock(obj)
            obj.Ticks = 0;
        end
        
        %% FUNCTION Get MO ID from name
        function moId = mdGetMoId(obj,vName)
            % Search for ID
            for idx = 1:length(obj.MoList)
                if(strcmp(obj.MoList(idx).name,vName))
                    moId = idx;
                    return
                end
            end
            moId = 0;
        end
        
        %% FUNCTION Set Job States
        function mdSetJobStates(obj,JobStates)
            obj.JobStates = JobStates;
        end
        
        %% FUNCTION Get Job States
        function JobStates = mdGetJobStates(obj)
            JobStates = obj.JobStates;
        end
        
        %% FUNCTION Process states
        function doRepeat = mdProcessStates(obj)
            doRepeat = true;
            while (doRepeat)
                doRepeat = false;
                for moId = 1:length(obj.MoList)
                    switch obj.MoList(moId).state
                        case obj.cInit % Initialized
                            % Do nothing
                        case obj.cReady
                            %[obj.MoList(moId).jobList.show] = deal(0);
                            %for jobId = 1:length(obj.MoList(moId).jobList)
                            %    obj.MoList(moId).jobList(jobId).show = 0;
                            %end
                            tmpCanStart = true;
                            for preId = 1:length(obj.MoList(moId).pre)
                                tmpPreMoId = obj.mdGetMoId(obj.MoList(moId).pre{preId});
                                tmpCanStart = tmpCanStart && ...
                                    obj.MoList(tmpPreMoId).state==obj.cDone;
                            end
                            for condId = 1:length(obj.MoList(moId).cond)
                                tmpCondMoId = obj.mdGetMoId(obj.MoList(moId).cond{condId});
                                tmpCanStart = tmpCanStart && ...
                                    obj.MoList(tmpCondMoId).state==obj.cDone;
                            end
                            if (tmpCanStart)
                                obj.MoList(moId).k = 0;
                                for jobId = 1:length(obj.MoList(moId).jobList)
                                    % If resynthesis on MO level is needed
                                    
                                    if (obj.bResynthMo)
                                        obj.mdSynthesizeJobStr(moId,jobId,1,1,1,1,0);
                                    end
                                    obj.JobStates{moId,jobId} = ...
                                        [2 obj.MoList(moId).jobList(jobId).droplet];
                                    obj.MoList(moId).jobList(jobId).show = 1;
                                end
                                for preId = 1:length(obj.MoList(moId).pre)
                                    tmpPreMoId = obj.mdGetMoId(obj.MoList(moId).pre{preId});
                                    for preJobId = 1:length(obj.MoList(tmpPreMoId).jobList)
                                        %if(obj.MoList(tmpPreMoId).jobList(preJobId).show)
                                            obj.MoList(tmpPreMoId).jobList(preJobId).show=0;
                                        %    break;
                                        %end
                                    end
                                end 
                                obj.MoList(moId).state = obj.cBusy;
                                doRepeat = true;
                            end
                        case obj.cBusy
                            % Check if goal is reached
                            bIsDone = true;
                            for jobId = 1:length(obj.MoList(moId).jobList)
                                tmpC = obj.JobStates{moId,jobId}(2:5);
                                tmpG = obj.MoList(moId).jobList(jobId).goal;
                                bIsDone = bIsDone && ...
                                    (tmpC(1)>=tmpG(1)) && (tmpC(2)>=tmpG(2)) &&...
                                    (tmpC(3)<=tmpG(3)) && (tmpC(4)<=tmpG(4)) ;
                            end
                            if (bIsDone)
                                %obj.JobStates(moId,:) = {[2 -1 -1 -1 -1]};
                                obj.MoList(moId).state = obj.cDone;
                                if any(strcmp(obj.MoList(moId).type,{'Out','Dsc'}))
                                    obj.MoList(moId).jobList(1).show = 0;
                                end
                                doRepeat = true;
                            end
                        case obj.cDone
                            % Do nothing
                        otherwise
                            disp('Error!');
                    end
                end
            end
            % Update internal state
            if (all([obj.MoList.state]==obj.cDone))
                obj.State = obj.cDone;
            elseif (any([obj.MoList.state]==obj.cBusy))
                obj.State = obj.cBusy;
            else
                obj.State = obj.cReady;
            end
        end
        
        %% FUNCTION Get State
        function results = mdIsDone(obj)
            results = (obj.State == obj.cDone);
        end
        
        %% FUNCTION Get Actions
        function Actions = mdGetActions(obj)
            for moId = 1:length(obj.MoList)
                if (obj.MoList(moId).state==obj.cBusy)
                    for jobId = 1:obj.JobCount(moId) %length(obj.MoList(moId).jobList)
                        obj.Actions{moId,jobId} = obj.pfcnGetAction(...
                            moId,jobId,...
                            obj.JobStates{moId,jobId},obj.MoList(moId).k);
                    end
                elseif (obj.MoList(moId).state==obj.cDone)
                    for jobId = 1:obj.JobCount(moId) %length(obj.MoList(moId).jobList)
                        obj.Actions{moId,jobId} = {};
                    end
                else
                    for jobId = 1:obj.JobCount(moId) %length(obj.MoList(moId).jobList)
                        obj.Actions{moId,jobId} = {};
                    end
                end
            end
            Actions = obj.Actions;
        end
        
        %% FUNCTION Process All MOs
        function results = mdPreprocessAllMos(obj)
            results = false;
            for idx = 1:length(obj.MoList)
                obj.mdPreprocessMo(idx);
            end
        end
        
        %% FUNCTION Process MO
        function results = mdPreprocessMo(obj,moId)
            results = false;
            % Validate moId
            if(moId>size(obj.MoList,2))
                return
            end
            % Check type
            switch obj.MoList(moId).type
                case 'Dis' % Dispense
                    % No job required
                    obj.MoList(moId).time = 2;
                    obj.MoList(moId).jobList(1).id = 1;
                    obj.MoList(moId).jobList(1).name = 'J1';
                    obj.MoList(moId).jobList(1).droplet = obj.MoList(moId).droplets{1};
                    obj.MoList(moId).jobList(1).goal = obj.MoList(moId).locations{1};
                    obj.MoList(moId).jobList(1).hazard = obj.pfcnGetHazard(...
                        obj.MoList(moId).droplets{1},obj.MoList(moId).locations{1});
                    obj.MoList(moId).jobList(1).kmax = 0;
                    obj.MoList(moId).jobList(1).drSize = fcnDrGetSize(...
                        obj.MoList(moId).jobList(1).droplet);
                    obj.MoList(moId).jobList(1).show = 0;
                    obj.MoList(moId).state = obj.cInit;
                    obj.JobCount(moId) = 1;
                case {'Out','Dsc'}
                    obj.MoList(moId).time = 2;
                    obj.MoList(moId).jobList(1).id = 1;
                    obj.MoList(moId).jobList(1).name = 'J1';
                    obj.MoList(moId).jobList(1).droplet = obj.MoList(moId).droplets{1};
                    obj.MoList(moId).jobList(1).goal = obj.MoList(moId).locations{1};
                    obj.MoList(moId).jobList(1).hazard = obj.pfcnGetHazard(...
                        obj.MoList(moId).droplets{1},obj.MoList(moId).locations{1});
                    obj.MoList(moId).jobList(1).kmax = 0;
                    obj.MoList(moId).jobList(1).drSize = fcnDrGetSize(...
                        obj.MoList(moId).jobList(1).droplet);
                    obj.MoList(moId).jobList(1).show = 0;
                    obj.MoList(moId).state = obj.cInit;
                    obj.JobCount(moId) = 1;
                    
                case {'Mix','Dlt'} % Mixing or Diluting
                    % 3 jobs required
                    for jobId=1:2
                        obj.MoList(moId).jobList(jobId).id = jobId;
                        obj.MoList(moId).jobList(jobId).name = ['J',int2str(jobId)];
                        obj.MoList(moId).jobList(jobId).droplet = ...
                            obj.MoList(moId).droplets{jobId};
                        obj.MoList(moId).jobList(jobId).goal = ...
                            obj.MoList(moId).locations{jobId};
                        obj.MoList(moId).jobList(jobId).hazard = obj.pfcnGetHazard(...
                            obj.MoList(moId).droplets{jobId},...
                            obj.MoList(moId).locations{jobId});
                        obj.MoList(moId).jobList(jobId).kmax = obj.pfcnGetKmax(...
                            obj.MoList(moId).jobList(jobId).droplet,...
                            obj.MoList(moId).jobList(jobId).goal);
                        obj.MoList(moId).jobList(jobId).show = 0;
                    end
                    obj.MoList(moId).state = obj.cInit;
                    obj.JobCount(moId) = 2;
                    
                case {'Mag','Was','Thm'} % Magnetic Sense
                    obj.MoList(moId).jobList(1).id = 1;
                    obj.MoList(moId).jobList(1).name = 'J1';
                    obj.MoList(moId).jobList(1).droplet = obj.MoList(moId).droplets{1};
                    obj.MoList(moId).jobList(1).goal = obj.MoList(moId).locations{1};
                    obj.MoList(moId).jobList(1).hazard = obj.pfcnGetHazard(...
                        obj.MoList(moId).droplets{1},obj.MoList(moId).locations{1});
                    obj.MoList(moId).jobList(1).kmax = obj.pfcnGetKmax(...
                        obj.MoList(moId).jobList(1).droplet,...
                        obj.MoList(moId).jobList(1).goal);
                    obj.MoList(moId).jobList(1).show = 0;
                    obj.MoList(moId).state = obj.cInit;
                    obj.JobCount(moId) = 1;
                    
                case {'Spt'} % Split
                    for jobId=1:2
                        obj.MoList(moId).jobList(jobId).id = jobId;
                        obj.MoList(moId).jobList(jobId).name = ['J',int2str(jobId)];
                        obj.MoList(moId).jobList(jobId).droplet = ...
                            obj.MoList(moId).droplets{jobId};
                        obj.MoList(moId).jobList(jobId).goal = ...
                            obj.MoList(moId).locations{jobId};
                        obj.MoList(moId).jobList(jobId).hazard = obj.pfcnGetHazard(...
                            obj.MoList(moId).droplets{jobId},...
                            obj.MoList(moId).locations{jobId});
                        obj.MoList(moId).jobList(jobId).kmax = obj.pfcnGetKmax(...
                            obj.MoList(moId).jobList(jobId).droplet,...
                            obj.MoList(moId).jobList(jobId).goal);
                        obj.MoList(moId).jobList(jobId).show = 0;
                    end
                    obj.MoList(moId).state = obj.cInit;
                    obj.JobCount(moId) = 2;
                    
                otherwise
                    fprintf('Unknown type %s!\n',obj.MoList(moId).type);
                    return
            end
            results = true;
        end
        
        %% FUNCTION Process All Jobs
        function results = mdProcessAllJobs(obj,bReSynth,bGenModel,bDStep,bLoadTb,bShortest)
            for moId = 1:length(obj.MoList)
                for jobId = 1:length(obj.MoList(moId).jobList)
                    obj.mdSynthesizeJobStr(moId,jobId,bReSynth,bGenModel,bDStep,bLoadTb,bShortest);
                end
                obj.MoList(moId).state = obj.cReady;
            end
            results = true;
        end
        
        %% FUNCTION Process Job
        function mdSynthesizeJobStr(obj,moId,jobId,bReSynth,bGenModel,bDStep,bLoadTb,bShortest)
            %TODO
            tic
            
            switch(obj.MoList(moId).type)
                case 'Dis'
                    obj.MoList(moId).jobList(jobId).pMaxVal = 1;
                otherwise
                    % Health matrix
                    if(bShortest)
                        tmpHealthMatrix = (2^obj.Bits-1)*ones(size(obj.HealthMatrix));
                    else
                        tmpHealthMatrix = obj.HealthMatrix;
                    end
                    % Current job
                    tmpJob = obj.MoList(moId).jobList(jobId);
                    % Check if strategy already exists
                    [bExist,UniqueID] = fcnLookupStrategy(...
                        obj.sPostFix,...
                        tmpJob.droplet,tmpJob.goal,tmpJob.hazard,tmpHealthMatrix);
                    % If no strategy exists and resynth required
                    if ((bExist==0 && bReSynth) || obj.bForceResynth)
                        VarList = {...
                            'cInitXa',int2str(tmpJob.droplet(1));...
                            'cInitYa',int2str(tmpJob.droplet(2));...
                            'cInitXb',int2str(tmpJob.droplet(3));...
                            'cInitYb',int2str(tmpJob.droplet(4));...
                            'cHazXa' ,int2str(tmpJob.hazard(1));...
                            'cHazYa' ,int2str(tmpJob.hazard(2));...
                            'cHazXb' ,int2str(tmpJob.hazard(3));...
                            'cHazYb' ,int2str(tmpJob.hazard(4));...
                            'cGoalXa',int2str(tmpJob.goal(1));...
                            'cGoalYa',int2str(tmpJob.goal(2));...
                            'cGoalXb',int2str(tmpJob.goal(3));...
                            'cGoalYb',int2str(tmpJob.goal(4));...
                            'Kmax',   int2str(tmpJob.kmax)};
                        % Check if double step model required
                        if(bDStep)
                            tmpModelNM = 'MEDAX';
                        else
                            tmpModelNM = 'MEDAY';
                        end
                        % Add model type
                        tmpModelNM = [tmpModelNM,'_',obj.ModelType];
                        tmpPropsNM = ['MEDA','_',obj.ModelType];
                        % Check if model generation is required
                        if(bGenModel)
                            fcnGenerateModel(obj.ModelType,0,...
                                obj.Width,obj.Height,...
                                tmpJob.hazard,tmpHealthMatrix,obj.Bits,bDStep);
                        end
                        [~,~] = obj.pfcnRunPrism(tmpModelNM,tmpPropsNM,...
                            obj.MoList(moId).name,...
                            obj.MoList(moId).jobList(jobId).name,...
                            VarList,...
                            UniqueID);
                        %disp(tmpY);
                    end
                    % Parse output files
                    %UniqueID = '';
                    [obj.MoList(moId).jobList(jobId).strTB,...
                        obj.MoList(moId).jobList(jobId).pMaxVal] = ...
                        obj.pfcnParseStrFile(moId,jobId,bLoadTb,UniqueID);
            end
            tmpTime = toc;
            fprintf('\t\t%s.%s %s, pmax=%1.5f (%1.3fs)\n',...
                obj.MoList(moId).name,...
                obj.MoList(moId).jobList(jobId).name,...
                obj.MoList(moId).type,...
                obj.MoList(moId).jobList(jobId).pMaxVal,...
                tmpTime);
        end
        
        %% FUNCTION Get Hazard Zone
        function hazard = pfcnGetHazard(obj,vDroplet,vLocation)
            %tmpA = droplets';
            %tmpB = locations';
            tmpAll = [vDroplet;vLocation];
            hazard = [min(tmpAll(:,1:2))-2,max(tmpAll(:,3:4))+2];
        end
        
        %% FUNCTION Parse Strategy File
        function [StrTable,pMaxVal] = pfcnParseStrFile(obj,moId,jobId,bLoadTb,UniqueID)
            % Files
            uniqueName = [obj.OutputPath,'\',...
                UniqueID];
            strFileName = [uniqueName,'_','adv.txt'];
            staFileName = [uniqueName,'_','sta.txt'];
            resFileName = [uniqueName];
            if (bLoadTb)
                if(strcmp(obj.ModelType,'SMG'))
                    % Strategy File
                    StrTB = readtable(strFileName,...
                        'FileType','text',...
                        'Delimiter',{',',':'});
                    StrTB.Var4 = [];
                    StrTB.Properties.VariableNames = {'vSID','vK','vActName'};
                    % State File
                    StaTB = readtable(staFileName,...
                        'FileType','text',...
                        'HeaderLines', 1,...
                        'ConsecutiveDelimitersRule', 'join',...
                        'Delimiter',{':','[',']',','});
                    StaMat = table2array(StaTB);
                    % Cross Reference
                    vState = nan(size(StrTB,1),5);
                    for i = 1:size(StrTB,1)
                        tmpRow = StaMat(:,1)==StrTB.vSID(i);
                        vState(i,:) = StaMat(tmpRow,2:6);
                    end
                    StrTB = [table(vState),StrTB];
                    % Return results
                    StrTable = StrTB;
                    % Note: to obtain value from StrTB:
                    % find(ismember(StrTB.vState,x,'rows') & StrTB.vK==6 )
                elseif (strcmp(obj.ModelType,'MDP'))
                    %TODO
                    StrTB = readtable(strFileName,...
                        'FileType','text',...
                        'Delimiter',{' '});
                    if(size(StrTB,2)==4)
                        StrTB.Var2 = [];
                        StrTB.Var3 = [];
                        StrTB.Properties.VariableNames = {'vSID','vActName'};
                        % State File
                        StaTB = readtable(staFileName,...
                            'FileType','text',...
                            'HeaderLines', 1,...
                            'ConsecutiveDelimitersRule', 'join',...
                            'Delimiter',{':','[',']',','});
                        StaMat = table2array(StaTB);
                        % Cross Reference
                        vState = nan(size(StrTB,1),5);
                        for i = 1:size(StrTB,1)
                            tmpRow = StaMat(:,1)==StrTB.vSID(i);
                            vState(i,:) = StaMat(tmpRow,2:6);
                        end
                        StrTB = [table(vState),StrTB];
                        % Return results
                        StrTable = StrTB;
                        % Note: to obtain value from StrTB:
                        % find(ismember(StrTB.vState,x,'rows') & StrTB.vK==6 )
                    else
                        fprintf('Warning: %s.%s no strategy found\n',...
                            obj.MoList(moId).name,...
                            obj.MoList(moId).jobList(jobId).name);
                        StrTable = [];
                    end
                else
                    StrTable = [];
                end
                % Results file
                fID = fopen(resFileName,'r');
                ResCells = textscan(fID,'%s');
                pMaxVal = str2double(ResCells{1}{2});
                fclose(fID);
            else
                StrTable = [];
                pMaxVal = NaN;
            end
        end
        
        %% FUNCTION Get action from job
        function Action = pfcnGetAction(obj,moId,jobId,vState,vK)
            if (~isempty(obj.MoList(moId).jobList(jobId).strTB))
                switch obj.ModelType
                    case 'SMG'
                        vK = min(vK,obj.MoList(moId).jobList(jobId).kmax);
                        rowIdx = ...
                            all(obj.MoList(moId).jobList(jobId).strTB.vState==vState,2) & ...
                            obj.MoList(moId).jobList(jobId).strTB.vK==vK ;
                        actionName = obj.MoList(moId).jobList(jobId).strTB.vActName(rowIdx);
                        if isempty(actionName)
                            fprintf('Warning: %s.%s could not find state %s\n',...
                                obj.MoList(moId).name,...
                                obj.MoList(moId).jobList(jobId).name,...
                                int2str(vState));
                            actionName = '';
                        else
                            actionName = actionName{:};
                        end
                        actionData = [];
                    case 'MDP'
                        %TODO
                        rowIdx = ...
                            all(obj.MoList(moId).jobList(jobId).strTB.vState==vState,2);
                        actionName = obj.MoList(moId).jobList(jobId).strTB.vActName(rowIdx,1);
                        if ( isempty(actionName) &&...
                                all(obj.MoList(moId).jobList(jobId).goal==vState(2:5)) )
                            actionName = 'aH';
                        elseif ( isempty(actionName) )
                            fprintf('Warning: %s.%s could not find state %s\n',...
                                obj.MoList(moId).name,...
                                obj.MoList(moId).jobList(jobId).name,...
                                int2str(vState));
                            actionName = '';
                        else
                            actionName = actionName{1};
                        end
                        actionData = [];
                end
            elseif (strcmp(obj.MoList(moId).type,'Dis'))
                actionName = 'aDispense';
                tmpDelta = obj.MoList(moId).jobList(jobId).goal - vState(2:5);
                tmpDelta = sign(tmpDelta).*min(abs(tmpDelta),[2 2 2 2]);
                actionData = vState(2:5)+tmpDelta;
            else
                %fprintf('Warning: no strTB found for %s.%s\n',...
                %    obj.MoList(moId).name, obj.MoList(moId).jobList(jobId).name);
                actionName = 'aH';
                actionData = [];
            end
            Action = {actionName,actionData};
        end
        
        %% FUNCTION Run Prism
        function [tmpX,tmpY] = pfcnRunPrism(...
                obj,vModelFN,vPropFN,vMoName,vJobName,vVars,vUniqueID)
            % Store current path
            tmpOriginalPath = pwd;
            cd(obj.PrismBinPath);
            % Objective ID
            objId = int2str(1); % q index starts at 1
            % Create script name
            uniqueName = [vUniqueID];
            scriptName = [ vModelFN,'_',vMoName,'_',vJobName,'.bat' ];
            %fprintf('%s\t Creating... ',scriptName);
            % Create script contents
            fID = fopen(scriptName,'w');
            fprintf(fID,'SET MODEL_FILE=%s.prism\n',vModelFN);
            fprintf(fID,'SET PROPS_FILE=%s.props\n',vPropFN);
            fprintf(fID,'SET ADV_FILE=%s\n',uniqueName);
            fprintf(fID,'SET PVEC_FILE=%s%s.m\n',uniqueName,'_pvec');
            fprintf(fID,'SET PROP_ID=%s\n',objId);
            fprintf(fID,'%s%s\n','SET PROJ_DIR=',obj.ProjectPath);
            fprintf(fID,'%s%s\n','SET OUT_DIR=',obj.OutputPath);
            tmpStr = '';
            for i = 1:length(vVars)
                if (~isempty(tmpStr))
                    tmpStr = [tmpStr,','];
                end
                tmpStr = [tmpStr, sprintf('%s=%s',vVars{i,1},vVars{i,2})];
            end
            fprintf(fID, 'SET OP_CONST=%s\n', tmpStr);
            fprintf(fID,'%s\n', [ ...
                'prism ' ...
                '%PROJ_DIR%\%MODEL_FILE% ' ...
                '%PROJ_DIR%\%PROPS_FILE% -explicit ' ...
                '-const %OP_CONST% ' ...
                '-prop %PROP_ID% ' ...
                '-exportstrat %OUT_DIR%\%ADV_FILE%_adv.txt:type=actions ' ...
                '-exportstates %OUT_DIR%\%ADV_FILE%_sta.txt ' ...
                '-exportresults %OUT_DIR%\%ADV_FILE% ' ...
                ...%'-v' ...
                ]);
            %'-exportstates %OUT_DIR%\%ADV_FILE%_sta.txt ' ...
            fclose(fID);
            % Run script
            fprintf('\t... ');
            [tmpX, tmpY] = system(scriptName);
            fprintf('Code %d\t',tmpX);
            if (tmpX ~=0)
                tmpY
                cd(tmpOriginalPath);
                ME = MException('Synth:PrismError', ...
                    'Script %s failed -- are you in the right directory?', ...
                    scriptName);
                throw(ME)
            end
            % Create log file
            fID = fopen(['logs\',scriptName,'.log'],'w');
            fprintf(fID,'%s',tmpY);
            fclose(fID);
            cd(tmpOriginalPath)
        end
        
        %% FUNCTION Estimate kmax
        function kMax = pfcnGetKmax(obj,dr1,dr2)
            tmpDist = fcnDrGetDistance(dr1,dr2);
            kMax = ceil(tmpDist*1.5)+1;
        end
    end
    
    
    
    
    
    
    
    
    
    
    
    
end