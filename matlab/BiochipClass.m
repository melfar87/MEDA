classdef BiochipClass < handle & matlab.System
% BIOCHIPCLASS Create BiochipClass object
%   BC = BiochipClass(HEIGHT,WIDTH) creates a biochip object of size
%   HEIGHTxWIDTH.
%
% Author: Mahmoud ELFAR
% email: mahmoud.elfar@duke.edu
% 2020-06-13

    %% Read-only properties
    properties (SetAccess=public, GetAccess=public)
        % Physical properties and features
        Height {mustBeInteger,mustBePositive}
        Width {mustBeInteger,mustBePositive}
        Dispensers
        % Actuation
        Actuators
        ActuationPatterns
        GlobalPattern
        IsActive {mustBeInteger}
        HealthMatrix
        MarkerSize = 200
        ActionSet = {...
            'aN','aS','aE','aW','aNE','aNW','aSE','aSW','aNN','aSS','aEE','aWW',...
            'aXbpYbm','aXbpYap','aXamYbm','aXamYap',...
            'aYbpXbm','aYbpXap','aYamXbm','aYamXap',...
            'aF','aH','aCon','aTrm','aEnd'}
        %MoveSet =  {'Z',...
        %    'N','S','E','W',...
        %    'NN','SS','EE','WW',...
        %    'NE','NW','SE','SW',...
        %    'aXbpYbm','aXbpYap','aXamYbm','aXamYap',...
        %    'aYbpXbm','aYbpXap','aYamXbm','aYamXap'}
        TransSet = {[0 0 0 0],...
            [0 1 0 1],[0 -1 0 -1],[1 0 1 0],[-1 0 -1 0],...
            [0 2 0 2],[0 -2 0 -2],[2 0 2 0],[-2 0 -2 0],...
            [1 1 1 1],[-1 1 -1 1],[1 -1 1 -1],[-1 -1 -1 -1],...
            [0 0 1 -1],[0 1 1 0],[-1 0 0 -1],[-1 1 0 0],...
            [0 0 -1 1],[1 0 0 1],[0 -1 -1 0],[1 -1 0 0]}
            
        % Sensing
        Actions
        Sensors
        Ticks = 0
        JobStates % Droplet locations
        PatternStates % Pattern locations
    end
    
    %% Private Properties
    properties (Access=public)
        DispensingJobs
        Droplets
        % Degradation matrix [0,1]^size
        Taus % degradation coefficient
        CVal = 1000 % degradation constant
        C1s
        C2s
        Degradation % Degradation matrix, each value is in [0 1]
        N % number of activations
        Bits % number of bits
        % Handles
        FigureHandle % Figure handler for biochip figure
        AxesHandle % Axes handler for the biochip figure
        MchScatterHandle % Scatter plot handler for MC health
        MchColors % Colors for MchScatterHandle
        DropletHandles % Handles for droplets
        PatternHandles % Handles for Actuations
        ScHandle % Handle for the scheduler
        % Options
        bSkipPlots
        bUseFixedHealth
    end
    
    
    
    %% Constructor
    methods
        function obj = BiochipClass(...
                Height,Width,HBits,...
                TauRange,C1Range,C2Range,dPerc,sDegMode,...
                bFixedH)
            % Initialize properties
            obj.Height = Height;
            obj.Width = Width;
            obj.Bits  = HBits;
            obj.bUseFixedHealth = bFixedH;
            % Initialize chip
            obj.Dispensers = struct(...
                'id', {}, 'xpos', {}, 'ypos', {}, 'type', {});
            obj.Actuators = zeros(obj.Height,obj.Width);
            obj.ActuationPatterns = struct(...
                'id', {}, 'droplet', {}, 'pattern', {}, 'isActive', {}, 'action', {});
            obj.IsActive = false;
            obj.Sensors = zeros(obj.Height,obj.Width);
            obj.DispensingJobs = struct(...
                'id', {}, 'isBusy', {}, 'radius', {});
            obj.Droplets = struct(...
                'id', {}, 'shape', {});
            obj.GlobalPattern = zeros(obj.Height,obj.Width);
            % Initialize job states
            obj.JobStates = cell(0,0);
            % Initialize number of activations
            obj.N = zeros(obj.Height,obj.Width);
            % Load health matrix
            if (obj.bUseFixedHealth)
                obj.Degradation = fcnLoadHealthMatrix();
                obj.Taus = ones(obj.Height,obj.Width);
            else
                obj.Taus = ...
                    TauRange(1,1)+(TauRange(1,2)-TauRange(1,1))*rand(obj.Height,obj.Width);
                obj.C1s = C1Range(1,1)*ones(obj.Height,obj.Width);
                obj.C2s = C2Range(1,1)*ones(obj.Height,obj.Width);
                if (dPerc>0)
                    switch sDegMode
                        case 'Clustered'
                            tmpClusters = datasample(...
                                1:(obj.Width*obj.Height)/4,...
                                round(dPerc*obj.Width*obj.Height/4),...
                                'Replace',false);
                            tmpIdxMatrix = zeros(obj.Height,obj.Width);
                            for tmpClusterIdx = tmpClusters
                                [tmpRow,tmpCol] = ind2sub(...
                                    [obj.Height/2,obj.Width/2],...
                                    tmpClusterIdx);
                                tmpRow = tmpRow*2;
                                tmpCol = tmpCol*2;
                                tmpIdxMatrix(tmpRow-1:tmpRow,tmpCol-1:tmpCol) = 1;
                            end
                            tmpIdx = find(tmpIdxMatrix);        
                            obj.Taus(tmpIdx) = ...
                                TauRange(2,1)+(TauRange(2,2)-TauRange(2,1))...
                                *rand(1, length(tmpIdx));
                            obj.C1s(tmpIdx) = C1Range(2,1);
                            obj.C2s(tmpIdx) = C2Range(2,1);
                        case 'Scattered'
                            tmpIdx = datasample(...
                                1:(obj.Width*obj.Height),...
                                round(dPerc*obj.Width*obj.Height),...
                                'Replace',false);
                            obj.Taus(tmpIdx) = ...
                                TauRange(2,1)+(TauRange(2,2)-TauRange(2,1))...
                                *rand(1, length(tmpIdx));
                            obj.C1s(tmpIdx) = C1Range(2,1);
                            obj.C2s(tmpIdx) = C2Range(2,1);
                    end
                end
                obj.Degradation = obj.Taus.^((obj.N+obj.C1s)./obj.C2s);
            end
            obj.pfcnUpdateHealthMatrix();
            % Data for plottting
            tmpX = 1:obj.Width;
            tmpY = 1:obj.Height;
            [tmpX,tmpY] = meshgrid(tmpX,tmpY);
            tmpZ = repmat(obj.MarkerSize,length(tmpX(:)),1);
            obj.MchColors = repmat(reshape(obj.Degradation,[],1),1,3);
            %obj.MchColors = repmat([0 0 0],length(tmpX(:)),1);
            %tmpZ = linspace(0.2,3,length(tmpX(:)))*obj.MarkerSize;
            % Initialize Biochip figure
            obj.FigureHandle = figure(...
                'Units', 'Normalized', 'OuterPosition', [1,0.05,0.9,0.9],...
                'Visible','off');
            obj.AxesHandle = gca;
            set(obj.AxesHandle,'XLimMode','manual','YLimMode','manual');
            hold(obj.AxesHandle,'on');
            axis equal; axis([0 obj.Width+1 0 obj.Height+1]);
            box(obj.AxesHandle,'on');
            % Initialize MchScatterHandle
            obj.MchScatterHandle = scatter(tmpX(:),tmpY(:),tmpZ(:),...
                obj.MchColors,'s','filled',...
                'MarkerEdgeColor',[0.7 0.7 0.7]);
        end
    end
    
    %% Public Synchronous Methods
    methods (Access=public)
        %% FUNCTION Refresh control pattern
        function amdRefreshActuatorsCb(obj)
            pvtConstructActuationPattern(obj);
        end
        
        %% FUNCTION Set Mode
        function amdSetOptions(obj,bSkipPlots)
            obj.bSkipPlots = bSkipPlots;
        end
           
        %% FUNCTIOn Set Scheduler
        function amdSetScheduler(obj,ScObj)
            obj.ScHandle = ScObj;
        end
        
        %% FUNCTION Initialize Job States
        function JobStates = amdInitJobStates(obj,rows,cols)
            % Initialize JobStates and PatternStates
            obj.JobStates = cell(rows,cols);
            obj.JobStates(:,:) = {[2 -1 -1 -1 -1]};
            obj.PatternStates = cell(rows,cols);
            obj.PatternStates(:,:) = {[-1 -1 -1 -1]};
            JobStates = obj.JobStates;
            % Initialize Plots
            obj.DropletHandles = cell(rows,cols,2);
            obj.PatternHandles = cell(rows,cols);
            for moId = 1:length(obj.ScHandle.MoList)
                for jobId = 1:length(obj.ScHandle.MoList(moId).jobList)
                    obj.amdUpdateDropletPlot(moId,jobId);
                    obj.amdUpdatePatternPlot(moId,jobId);
                end
            end
        end
        
        %% FUNCTION Update Droplet Plot
        function amdUpdateDropletPlot(obj,moId,jobId)
            if (obj.bSkipPlots)
                return
            end
            if (obj.ScHandle.MoList(moId).jobList(jobId).show)
                tmpDr = obj.JobStates{moId,jobId}(2:5) + [-0.6 -0.6 0.6 0.6];
                tmpX = [tmpDr(1),tmpDr(3),tmpDr(3),...
                    tmpDr(1),tmpDr(1)];
                tmpY = [tmpDr(2),tmpDr(2),tmpDr(4),...
                    tmpDr(4),tmpDr(2)];
                tmpT = sprintf('%s%s[%02d.%01d]',...
                    obj.ScHandle.MoList(moId).name,...
                    obj.ScHandle.MoList(moId).jobList(jobId).name,...
                    moId,jobId);
                if (isempty(obj.DropletHandles{moId,jobId,1}))
                    % Create
                    obj.DropletHandles{moId,jobId,1} = plot(obj.AxesHandle,...
                        tmpX,tmpY,'Color',[0 0.4470 0.7410],'LineWidth',2);
                    obj.DropletHandles{moId,jobId,2} = text(obj.AxesHandle,...
                        tmpX(1),tmpY(4)+0.4,-1,tmpT);
                    obj.DropletHandles{moId,jobId,2}.Color = 'white';
                    obj.DropletHandles{moId,jobId,2}.BackgroundColor = [0 0.4470 0.7410];
                    obj.DropletHandles{moId,jobId,2}.Margin = 1;
                else
                    % Update
                    obj.DropletHandles{moId,jobId,1}.XData = tmpX;
                    obj.DropletHandles{moId,jobId,1}.YData = tmpY;
                    obj.DropletHandles{moId,jobId,2}.String = tmpT;
                    obj.DropletHandles{moId,jobId,2}.Position = [tmpX(1),tmpY(4)+0.4];
                end
            elseif (~isempty(obj.DropletHandles{moId,jobId,1}))
                delete(obj.DropletHandles{moId,jobId,1});
                obj.DropletHandles{moId,jobId,1} = [];
                delete(obj.DropletHandles{moId,jobId,2});
                obj.DropletHandles{moId,jobId,2} = [];
            else
                % Do nothing
            end
                
        end
        
        %% FUNCTION Update Pattern Scatter
        function amdUpdatePatternPlot(obj,moId,jobId)
            if (obj.bSkipPlots)
                return
            end
            if (obj.ScHandle.MoList(moId).jobList(jobId).show)
                tmpPattern = obj.PatternStates{moId,jobId};
                tmpX = tmpPattern(1):tmpPattern(3);
                tmpY = tmpPattern(2):tmpPattern(4);
                [tmpX,tmpY] = meshgrid(tmpX,tmpY);
                tmpZ = repmat(obj.MarkerSize+50,length(tmpX(:)),1);
                if (isempty(obj.PatternHandles{moId,jobId}))
                    % Create
                    obj.PatternHandles{moId,jobId} = scatter(obj.AxesHandle,...
                        tmpX(:),tmpY(:),tmpZ(:),'s',...
                        'LineWidth',1,'MarkerEdgeColor',[0.8500 0.3250 0.0980]);
                else
                    % Update
                    obj.PatternHandles{moId,jobId}.XData = tmpX(:);
                    obj.PatternHandles{moId,jobId}.YData = tmpY(:);
                    obj.PatternHandles{moId,jobId}.ZData = tmpZ(:);
                end
            elseif (~isempty(obj.PatternHandles{moId,jobId,1}))
                delete(obj.PatternHandles{moId,jobId});
                obj.PatternHandles{moId,jobId} = [];
            else
                % do nothing
            end
        end
        
        %% FUNCTION Get Job States
        function JobStates = amdGetJobStates(obj)
            JobStates = obj.JobStates;
        end
        
        %% FUNCTION Set Job States
        function amdSetJobStates(obj,JobStates)
            obj.JobStates = JobStates;
        end
        
        %% FUNCTION Show Biochip
        function amdShowBiochip(obj)
            if(isvalid(obj.FigureHandle))
                obj.FigureHandle.Visible = 'on';
            end
        end
        
        %% FUNCTION Hide Biochip
        function amdHideBiochip(obj)
            if(isvalid(obj.FigureHandle))
                obj.FigureHandle.Visible = 'off';
            end
        end
    end
    
    %% Public Asynchronous Methods
    methods (Access=public)
        %% FUNCTION Add new dispenser
        %   - Returns new dispenser ID if successul, 0 otherwise
        function dispenserId = mdAddDispenser(obj, xpos, ypos, dispenserType)
            % Validate location
            if( (xpos<1) || (xpos>obj.Width) || (ypos<1) || (ypos>obj.Height) )
                % Invalid input
                dispenserId = 0;
                return
            end
            % Get new dispenser ID
            dispenserId = size(obj.Dispensers,2)+1; % I hate length()
            % Create new dispenser
            obj.Dispensers(dispenserId) = struct( ...
                'id', dispenserId, ...
                'xpos', 0, ...
                'ypos', 0, ...
                'type', dispenserType);
            % Create new dispenser job queue
            obj.DispensingJobs(dispenserId) = struct( ...
                'id', dispenserId, ...
                'isBusy', false, ...
                'radius', 0);
        end
        
        %% FUNCTION Dispense droplet
        function results = mdDispense(obj, dispenserId, dropletRadius)
            % Validate dispenserId
            if (dispenserId>size(obj.DispensingJobs,2))
                % Invalid input
                results = false;
                return
            end
            % Register to the queue
            if (~obj.DispensingJobs(dispenserId).isBusy)
                obj.DispensingJobs(dispenserId).radius = dropletRadius;
                obj.DispensingJobs(dispenserId).isBusy = true;
            else
                % Dispenser is busy, no queuing is supported
                results = false;
            end
        end
        
        %% FUNCTION Add new actuation pattern using coordinates
        function patternId = mdAddDroplet(obj,droplet,isActive)
            % Validate doEnable
            if ( (isActive~=0) && (isActive~=1) )
                patternId = 0;
                return
            end
            % Validate droplet size
            if (~isequal(size(droplet),[1 4]))
                patternId = 0;
                return
            end
            % Validate droplet value
            if (droplet(1)<1 || droplet(2)<1 || droplet(3)>obj.Width || droplet(4)>obj.Height)
                patternId = 0;
                return
            end
            % Generate actuation pattern
            actuationPattern = zeros(obj.Height,obj.Width);
            actuationPattern(droplet(2):droplet(4),droplet(1):droplet(3)) = 1;
            % Get new actuation pattern ID
            patternId = size(obj.ActuationPatterns,2)+1;
            % Create new pattern
            obj.ActuationPatterns(patternId) = struct( ...
                'id', patternId, ...
                'droplet', droplet, ...
                'pattern', logical(actuationPattern), ...
                'isActive', isActive, ...
                'action', []);
        end
        
        %% FUNCTION Add new actuation pattern
        function patternId = mdAddActuationPattern(obj,actuationPattern,isActive)
            % Validate doEnable
            if ( (isActive~=0) && (isActive~=1) )
                patternId = 0;
                return
            end
            % Validate actuationPattern
            if (~isequal(size(actuationPattern),[obj.Height,obj.Width]))
                patternId = 0;
                return
            end
            % Get new actuation pattern ID
            patternId = size(obj.ActuationPatterns,2)+1;
            % Create new pattern
            obj.ActuationPatterns(patternId) = struct( ...
                'id', patternId, ...
                'droplet', {}, ...
                'pattern', logical(actuationPattern), ...
                'isActive', isActive, ...
                'action', []);
        end
        
        %% FUNCTION Move an actuation pattern
        function results = mdSetAction(obj,patternId,action)
            % Validate patternId
            if(patternId>size(obj.ActuationPatterns,2))
                results = false;
                return
            end
            % Validate action
            if(~pvtIsActionValid(obj,{action}))
                results = false;
                return
            end
            % Assign action
            obj.ActuationPatterns(patternId).action = action;
            results = true;
        end
        
        %% FUNCTION Enable or disable global actuations
        function results = mdSetGlobalActivation(obj,isActive)
            % Validate doEnable
            if ( (isActive~=0) && (isActive~=1) )
                results = false;
                return
            end
            % Set global actuation
            obj.IsActive = logical(isActive);
            results = true;
        end
        
        %% FUNCTION Apply Actions
        function mdApplyActions(obj,Actions)
            obj.Actions = Actions;
            for moId = 1:obj.ScHandle.MoCount %length(obj.ScHandle.MoList)
                for jobId = 1:obj.ScHandle.JobCount(moId) %length(obj.ScHandle.MoList(moId).jobList)
                    if (~isempty(obj.Actions{moId,jobId}))
                        actionName = obj.Actions{moId,jobId}{1};
                        actionData = obj.Actions{moId,jobId}{2};
                        if (~isempty(actionData)) % Probably mixing
                            droplet = actionData;
                            obj.JobStates{moId,jobId}(2:5) = droplet;
                            obj.PatternStates{moId,jobId} = droplet;
                        else
                            droplet = obj.JobStates{moId,jobId}(2:5);
                            [pmat, reqMoveId] = obj.pvtGetProb(droplet, actionName);
                            pd = makedist('Multinomial','probabilities',pmat);
                            moveId = random(pd);
                            % Pattern moves deterministically
                            obj.PatternStates{moId,jobId} = ...
                                droplet + obj.TransSet{reqMoveId};
                            % Droplets move is probabilistic
                            obj.JobStates{moId,jobId}(2:5) = ...
                                droplet + obj.TransSet{moveId};
                            %obj.ActuationPatterns(idx).pattern = ...
                            %    obj.pvtGetPattern(obj.ActuationPatterns(idx).droplet);
                            %obj.ActuationPatterns(idx).action = [];
                        end
                    %elseif (~obj.ScHandle.MoList(moId).jobList(jobId).show)
                    %    obj.JobStates{moId,jobId}(2:5) = [-1 -1 -1 -1];
                    %    obj.PatternStates{moId,jobId} = [-1 -1 -1 -1];
                    else
                        % Keep the hold pattern shown
                    end
                    % Update plots
                    if (~obj.bSkipPlots)
                        obj.amdUpdateDropletPlot(moId,jobId);
                        obj.amdUpdatePatternPlot(moId,jobId);
                    end
                end
            end
        end
        
        %% FUNCTION Degrade chip
        function mdDegrade(obj)
            if (obj.bUseFixedHealth)
                return;
            end
            % Update global pattern
            obj.pvtUpdateGlobalPattern();
            % Increment N
            obj.N = obj.N + obj.GlobalPattern;
            % Update degradation
            obj.Degradation = obj.Taus.^((obj.N+obj.C1s)./obj.C2s);
            % Update plot
            if (~obj.bSkipPlots)
                obj.pfcnUpdateColors();
            end
        end
        
        %% FUNCTION Read health matrix
        function HM = mdReadHealthMatrix(obj)
            obj.pfcnUpdateHealthMatrix();
            HM = obj.HealthMatrix;
        end
        
        function mdResetDegradation(obj,NVals)
            obj.N(:) = NVals;
            obj.Degradation = obj.Taus.^((obj.N+obj.C1s)./obj.C2s);
        end
        
        %% FUNCTION Update health matrix
        function pfcnUpdateHealthMatrix(obj)
            if (obj.bUseFixedHealth)
                obj.HealthMatrix = ...
                    ceil((max(obj.Degradation,0)-0)./(1-0)*(2^obj.Bits))-1;
            else
                obj.HealthMatrix = ...
                    ceil(obj.Degradation*(2^obj.Bits)) - 1;
                    %ceil((max(obj.Taus.^(obj.N/obj.CVal),0)-0)./(1-0)*(2^obj.Bits))-1;
            end
        end
        
    end
    
    
    
    %% Periodic Methods needed to simulate the chip
    methods (Access=public)
        %% FUNCTION Update Global Pattern
        function pvtUpdateGlobalPattern(obj)
            tmpGP = zeros(obj.Height,obj.Width);
            for moId = 1:length(obj.ScHandle.MoList)
                for jobId = 1:length(obj.ScHandle.MoList(moId).jobList)
                    if (obj.ScHandle.MoList(moId).jobList(jobId).show)
                        p = obj.PatternStates{moId,jobId};
                        tmpGP(...
                            max(1,p(2)):min(obj.Height,p(4)),...
                            max(1,p(1)):min(obj.Width,p(3))) = 1;
                    end
                end
            end
            obj.GlobalPattern = tmpGP;
        end
        
        %% FUNCTION Construct actuation pattern
        function pvtConstructActuationPattern(obj)
            tmpPattern = zeros(obj.Height,obj.Width);
            tmpCount = size(obj.ActuationPatterns,2); % I hate length
            for idx = 1:1:tmpCount
                if (obj.ActuationPatterns(idx).isActive)
                    tmpPattern = tmpPattern | obj.ActuationPatterns(idx).pattern;
                end
            end
            obj.Actuators = tmpPattern;
        end
        
        
        
        %% FUNCTION Process dispensing jobs
        function pvtProcessDispensingJobs(obj)
            %[TODO] Implement
            tmpCount = size(obj.DispensingJobs,2);
            for idx = 1:1:tmpCount
                if (obj.DispensingJobs(idx).isBusy)
                    tmpId = size(obj.Droplets);
                end
            end
        end
        
        %% FUNCTION Process pattern movement
        function pvtProcessActions(obj)
            tmpCount = size(obj.ActuationPatterns,2);
            for idx = 1:1:tmpCount
                if ((obj.ActuationPatterns(idx).isActive)&& ...
                        ~isempty(obj.ActuationPatterns(idx).action))
                    pmat = obj.pvtGetProb(obj.ActuationPatterns(idx).droplet, ...
                        obj.ActuationPatterns(idx).action);
                    pd = makedist('Multinomial','probabilities',pmat);
                    moveId = random(pd);
                    %moveNm = obj.MoveSet{moveId};
                    obj.ActuationPatterns(idx).droplet = ...
                        obj.ActuationPatterns(idx).droplet + obj.TransSet{moveId};
                    obj.ActuationPatterns(idx).pattern = ...
                        obj.pvtGetPattern(obj.ActuationPatterns(idx).droplet);
                    obj.ActuationPatterns(idx).action = [];
                end
            end
        end
            
    end
    
    %% Common Private Methods
    methods (Access=public)
        
        %% FUNCTION check if action exists
        function results = pvtIsActionValid(obj,action)
            results = any(cellfun(@isequal,...
                obj.ActionSet, repmat(action, size(obj.ActionSet))));
        end
        
        %% FUNCTION get pattern from droplet
        function ap = pvtGetPattern(obj,droplet)
            ap = zeros(obj.Height,obj.Width);
            ap(droplet(2):droplet(4),droplet(1):droplet(3)) = 1;
            ap = logical(ap);
        end
        
        %% FUNCTION get probability of moving
        function [pmat,reqMoveId] = pvtGetProb(obj,droplet,action)
            d = droplet;
            D = obj.Degradation;
            [xa,ya,xb,yb] = deal(d(1),d(2),d(3),d(4));
            [pN,pS,pE,pW] = deal(0);
            [pNN,pSS,pEE,pWW] = deal(0);
            [pNE,pNW,pSE,pSW] = deal(0);
            [pXbpYbm,pXbpYap,pXamYbm,pXamYap,pYbpXbm,pYbpXap,pYamXbm,pYamXap] = deal(0);
            pZ = 0;
            pmat = zeros(3,3);
            reqMoveId = 1;
            switch action
                case 'aH'
                    reqMoveId = 1;
                    pZ = 1; 
                case 'aN'
                    reqMoveId = 2;
                    pN = mean(D(yb+1,xa:xb),'all');
                    pZ = 1-pN;
                case 'aS'
                    reqMoveId = 3;
                    pS = mean(D(ya-1,xa:xb),'all');
                    pZ = 1-pS;
                case 'aE'
                    reqMoveId = 4;
                    pE = mean(D(ya:yb,xb+1),'all');
                    pZ = 1-pE;
                case 'aW'
                    reqMoveId = 5;
                    pW = mean(D(ya:yb,xa-1),'all');
                    pZ = 1-pW;
                    
                case 'aNN'
                    reqMoveId = 6;
                    tpN  = mean(D(yb+1,xa:xb),'all');
                    tpNN = mean(D(yb+2,xa:xb),'all');
                    pNN = tpN*tpNN;
                    pN  = tpN*(1-tpNN);
                    pZ  = 1-tpN;
                case 'aSS'
                    reqMoveId = 7;
                    tpS  = mean(D(ya-1,xa:xb),'all');
                    tpSS = mean(D(ya-2,xa:xb),'all');
                    pSS = tpS*tpSS;
                    pS  = tpS*(1-tpSS);
                    pZ  = 1-tpS;
                case 'aEE'
                    reqMoveId = 8;
                    tpE  = mean(D(ya:yb,xb+1),'all');
                    tpEE = mean(D(ya:yb,xb+2),'all');
                    pEE = tpE*tpEE;
                    pE  = tpE*(1-tpEE);
                    pZ  = 1-tpE;
                case 'aWW'
                    reqMoveId = 9;
                    tpW  = mean(D(ya:yb,xa-1),'all');
                    tpWW = mean(D(ya:yb,xa-2),'all');
                    pWW = tpW*tpWW;
                    pW  = tpW*(1-tpWW);
                    pZ  = 1-tpW;
                    
                case 'aNE'
                    reqMoveId = 10;
                    pNEN = mean(D(     yb+1,xa+1:xb+1),'all');
                    pNEE = mean(D(ya+1:yb+1,     xb+1),'all');
                    pNE = pNEN*pNEE;
                    pN  = pNEN*(1-pNEE);
                    pE  = (1-pNEN)*pNEE;
                    pZ  = (1-pNEN)*(1-pNEE);
                case 'aNW'
                    reqMoveId = 11;
                    pNWN = mean(D(     yb+1,xa-1:xb-1),'all');
                    pNWW = mean(D(ya+1:yb+1,xa-1     ),'all');
                    pNW = pNWN*pNWW;
                    pN  = pNWN*(1-pNWW);
                    pW  = (1-pNWN)*pNWW;
                    pZ  = (1-pNWN)*(1-pNWW);
                case 'aSE'
                    reqMoveId = 12;
                    pSES = mean(D(ya-1     ,xa+1:xb+1),'all');
                    pSEE = mean(D(ya-1:yb-1,     xb+1),'all');
                    pSE = pSES*pSEE;
                    pS  = pSES*(1-pSEE);
                    pE  = (1-pSES)*pSEE;
                    pZ  = (1-pSES)*(1-pSEE);
                case 'aSW'
                    reqMoveId = 13;
                    pSWS = mean(D(ya-1     ,xa-1:xb-1),'all');
                    pSWW = mean(D(ya-1:yb-1,xa-1     ),'all');
                    pSW = pSWS*pSWW;
                    pS  = pSWS*(1-pSWW);
                    pW  = (1-pSWS)*pSWW;
                    pZ  = (1-pSWS)*(1-pSWW);                   
                case 'aXbpYbm'
                    reqMoveId = 14;
                    pXbpYbm = mean(D(ya:yb-1,xb+1),'all');
                    pZ = (1-pXbpYbm);
                case 'aXbpYap'
                    reqMoveId = 15;
                    pXbpYap = mean(D(ya+1:yb,xb+1),'all');
                    pZ = (1-pXbpYap);
                case 'aXamYbm'
                    reqMoveId = 16;
                    pXamYbm = mean(D(ya:yb-1,xa-1),'all');
                    pZ = (1-pXamYbm);
                case 'aXamYap'
                    reqMoveId = 17;
                    pXamYap = mean(D(ya+1:yb,xa-1),'all');
                    pZ = (1-pXamYap);
                case 'aYbpXbm'
                    reqMoveId = 18;
                    pYbpXbm = mean(D(yb+1,xa:xb-1),'all');
                    pZ = (1-pYbpXbm);
                case 'aYbpXap'
                    reqMoveId = 19;
                    pYbpXap = mean(D(yb+1,xa+1:xb),'all');
                    pZ = (1-pYbpXap);
                case 'aYamXbm'
                    reqMoveId = 20;
                    pYamXbm = mean(D(ya-1,xa:xb-1),'all');
                    pZ = (1-pYamXbm);
                case 'aYamXap'
                    reqMoveId = 21;
                    pYamXap = mean(D(ya-1,xa+1:xb),'all');
                    pZ = (1-pYamXap);
                otherwise
                    fprintf('Error: %s action not implemented!\n',action);
                    pZ = 1;
            end
            pmat = [pZ, pN, pS, pE, pW, pNN, pSS, pEE, pWW, pNE, pNW, pSE, pSW,...
                pXbpYbm,pXbpYap,pXamYbm,pXamYap,pYbpXbm,pYbpXap,pYamXbm,pYamXap];
            if(~sum(pmat,'all')==1)
                disp('Error!');
                pmat = [1];
            end
        end
        
        %% FUNCTION Update Figure colors
        function pfcnUpdateColors(obj)
            if (obj.bSkipPlots)
                return
            end
            obj.MchColors = repmat(reshape(obj.Degradation,[],1),1,3);
            if (~isempty(obj.MchScatterHandle))
                obj.MchScatterHandle.CData = obj.MchColors;
            end
        end
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
end