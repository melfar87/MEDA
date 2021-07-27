# classdef MedaSchedulerClass < handle & matlab.System
# # MEDASCHEDULERCLASS Create BiochipClass object
# #   BC = BiochipClass(HEIGHT,WIDTH) creates a biochip object of size
# #   HEIGHTxWIDTH.
# #
# # Author: Mahmoud ELFAR
# # email: mahmoud.elfar@duke.edu
# # 2020-08-07
from dataclasses import dataclass, field

from matlab import *

class MedaBiochip:
    # Read-only properties
    sPostFix = ''
    PrismBinPath = r"D:\PRISM\prism-games-compo\prism\bin"
    OutputPath = r"D:\WS\PRISM\MEDA\Output"
    ProjectPath = r"D:\WS\PRISM\MEDA"
    cInit = 0
    cReady = 1
    cBusy = 2
    cWait = 3
    cDone = 4
    MoTypes = {'Dis', 'Out', 'Dsc', 'Mix', 'Dlt', 'Mag', 'Was', 'Spt'}
    
    @dataclass
    class MoListStruct:
        id: int
        name: str
        type: str
        pre: list
        droplets: list
        locations: list
        cond: list
        state: int
        k: int
        time: int
        jobList: list = field(default_factory=list)

    @dataclass
    class JobListStruct:
        id: int
        name: str
        droplet: list
        goal: list
        hazard: list
        kmax: int
        drSize: int
        strTB: str
        pMaxVal: float
        show: int

    kmax = 30

    # Public properties
    Height = None
    Width = None
    Bits = None
    HealthMatrix = None
    Actions = None
    JobStates = None
    MoList = []
    MoCount = None
    JobCount = None
    Schedule = None
    Ticks = None
    State = None
    ModelType = 'SMG'
    bResynthMo = 0
    bForceResynth = 1 # Controls whether resynth is forced

    # Constructor
    def __init__(self, sPostFix,Height,Width,HBits,bResynthMo):
        
        self.Height = Height
        self.Width = Width
        self.Bits = HBits
        self.MoList = self.MoListStruct
        self.Ticks = 0
        self.State = self.cInit
        self.MoCount = 0
        self.JobCount = 0
        self.bResynthMo = bResynthMo
        self.sPostFix = sPostFix
        self.OutputPath = [self.OutputPath, self.sPostFix]
        return

    # Public Asynchronous Methods

    # FUNCTION Reset State
    def mdResetState(self):
        [self.MoList.state] = deal(self.cReady)
        [self.MoList.k] = deal(-1)
        self.Actions = []
        self.JobStates = []
        self.Ticks = 0
        self.State = self.cInit
        for moObj in self.MoList:
            for jobObj in moObj.jobList:
                jobObj.show = 0

        return


    # FUNCTION Set Model Type
    def mdSetModelType(self,sModelType):
        if sModelType == 'SMG' or sModelType == 'MDP':
            self.ModelType = sModelType
        else:
            fprintf('Error: Unknown model type #s\n',sModelType)
        return

    # FUNCTION Add new MO
    def mdAddMo(self,vName,vType,vPre,vStartDr,vEndDr,vCond):
        """
        :param vName:
        :param vType:
        :param vPre:
        :param vStartDr:
        :param vEndDr:
        :param vCond:
        :return: MO ID
        """
        # Get new ID
        moId = len(self.MoList) + 1
        moObj = self.MoListStruct()
        # Set parameters
        self.MoList(moId).id   = moId
        self.MoList(moId).name = vName
        self.MoList(moId).type = vType
        self.MoList(moId).pre  = vPre
        self.MoList(moId).droplets = vStartDr
        self.MoList(moId).locations = vEndDr
        if(exist('vCond','var')):
            self.MoList(moId).cond = vCond
        else:
            self.MoList(moId).cond = cell(0,0)
        # end
        self.MoList(moId).state = self.cInit
        self.MoList(moId).k = -1
        self.MoList(moId).time = -1
        self.MoList(moId).jobList = self.JobListStruct
        self.MoCount = self.MoCount+1
        return moId
    # end

    # FUNCTION Increment clock
    def mdTick(self):
        for moObj in self.MoList:
            if ((moObj.k>=0) and (moObj.state==self.cBusy)):
                moObj.k = moObj.k + 1
        self.Ticks = self.Ticks + 1
        return

    # FUNCTION Reset clock
    def mdResetClock(self):
        self.Ticks = 0
        return

    # FUNCTION Get MO ID from name
    def mdGetMoId(self,vName):
        """
        :param vName:
        :return: MO ID
        """
        # Search for ID
        for moObj in self.MoList:
            if(strcmp(moObj.name,vName)):
                moId = moObj.id
                return moId
        moId = 0
        return moId

    # FUNCTION Set Job States
    def mdSetJobStates(self,JobStates):
        self.JobStates = JobStates
    # end

    # FUNCTION Get Job States
    def mdGetJobStates(self):
        """
        :return: JobStates
        """
        return self.JobStates
    # end

    # FUNCTION Process states
    def mdProcessStates(self):
        """
        :return: doRepeat a flag whether to repeat
        """
        doRepeat = True
        while (doRepeat):
            doRepeat = False
            for moObj in self.MoList:
                moId = moObj.id
                if moObj.state == self.cInit: # Initialized
                    pass # Do nothing
                elif moObj.state == self.cReady:
                    tmpCanStart = True
                    for preObj in moObj.pre:
                        tmpPreMoId = self.mdGetMoId(preObj)
                        tmpCanStart = tmpCanStart and \
                                      self.MoList[tmpPreMoId].state==self.cDone
                    # end
                    for condObj in moObj.cond:
                        tmpCondMoId = self.mdGetMoId(condObj)
                        tmpCanStart = tmpCanStart and \
                                      self.MoList[tmpCondMoId].state==self.cDone
                    # end
                    if (tmpCanStart):
                        moObj.k = 0
                        for jobObj in moObj.jobList:
                            # If resynthesis on MO level is needed
                            if (self.bResynthMo):
                                self.mdSynthesizeJobStr(moObj.id,jobObj.id,1,1,1,1,0)
                            # end
                            self.JobStates[moId][jobId] = [2,jobObj.droplet]
                            jobObj.show = 1
                        # end
                        for preName in moObj.pre:
                            tmpPreMoId = self.mdGetMoId(preName)
                            for preJobObj in self.MoList[tmpPreMoId].jobList:
                                    preJobObj.show = 0
                        moObj.state = self.cBusy
                        doRepeat = True
                    # end
                elif moObj.state == self.cBusy:
                    # Check if goal is reached
                    bIsDone = True
                    for jobObj in moObj.jobList:
                        tmpC = self.JobStates[moId][jobId][1:5] # 2:5
                        tmpG = jobObj.goal
                        bIsDone = bIsDone \
                                  and (tmpC[0]>=tmpG[0]) \
                                  and (tmpC[1]>=tmpG[1]) \
                                  and (tmpC[2]<=tmpG[2]) \
                                  and (tmpC[3]<=tmpG[3])
                    # end
                    if (bIsDone):
                        moObj.state = self.cDone
                        if strcmp(moObj.type,['Out','Dsc']):
                            moObj.jobList[1].show = 0
                        doRepeat = True
                    # end
                elif moObj.state == self.cDone:
                    pass # Do nothing
                else:
                    disp('Error!')

        # end
        # Update internal state
        if (all([self.MoList.state]==self.cDone)):
            self.State = self.cDone
        elif (any([self.MoList.state]==self.cBusy)):
            self.State = self.cBusy
        else:
            self.State = self.cReady

        return doRepeat


    # FUNCTION Get State
    def mdIsDone(self):
        """

        :param self:
        :return: results
        """
        results = (self.State == self.cDone)
        return results


    # FUNCTION Get Actions
    def mdGetActions(self):
        """

        :param self:
        :return: Actions
        """
        for moObj in self.MoList:
            if (moObj.state==self.cBusy):
                moId = moObj.id
                for jobObj in moObj.jobList:
                    jobId = jobObj.id
                    self.Actions[moId][jobId] = self.pfcnGetAction(
                        moId,jobId, self.JobStates[moId][jobId],moObj.k)
                # end
            elif (moObj.state==self.cDone):
                for jobObj in moObj.jobList:
                    jobId = jobObj.id
                    self.Actions[moId][jobId] = []
            else:
                for jobObj in moObj.jobList:
                    jobId = jobObj.id
                    self.Actions[moId][jobId] = []
        Actions = self.Actions
        return Actions
    # end

    # FUNCTION Process All MOs
    def mdPreprocessAllMos(self):
        """

        :return: results
        """
        results = False
        for moObj in self.MoList:
            self.mdPreprocessMo(moObj)
        # end
    # end

    # FUNCTION Process MO
    # def mdPreprocessMo(self,moId):
    #     results = False
    #     # Validate moId
    #     if(moId>size(self.MoList,2))
    #         return
    #     # end
    #     # Check type
    #     switch self.MoList(moId).type:
    #         case 'Dis': # Dispense
    #             # No job required
    #             self.MoList(moId).time = 2
    #             self.MoList(moId).jobList(1).id = 1
    #             self.MoList(moId).jobList(1).name = 'J1'
    #             self.MoList(moId).jobList(1).droplet = self.MoList(moId).droplets{1}
    #             self.MoList(moId).jobList(1).goal = self.MoList(moId).locations{1}
    #             self.MoList(moId).jobList(1).hazard = self.pfcnGetHazard(
    #                 self.MoList(moId).droplets{1},self.MoList(moId).locations{1})
    #             self.MoList(moId).jobList(1).kmax = 0
    #             self.MoList(moId).jobList(1).drSize = fcnDrGetSize(
    #                 self.MoList(moId).jobList(1).droplet)
    #             self.MoList(moId).jobList(1).show = 0
    #             self.MoList(moId).state = self.cInit
    #             self.JobCount(moId) = 1
    #         case {'Out','Dsc'}
    #             self.MoList(moId).time = 2
    #             self.MoList(moId).jobList(1).id = 1
    #             self.MoList(moId).jobList(1).name = 'J1'
    #             self.MoList(moId).jobList(1).droplet = self.MoList(moId).droplets{1}
    #             self.MoList(moId).jobList(1).goal = self.MoList(moId).locations{1}
    #             self.MoList(moId).jobList(1).hazard = self.pfcnGetHazard(
    #                 self.MoList(moId).droplets{1},self.MoList(moId).locations{1})
    #             self.MoList(moId).jobList(1).kmax = 0
    #             self.MoList(moId).jobList(1).drSize = fcnDrGetSize(
    #                 self.MoList(moId).jobList(1).droplet)
    #             self.MoList(moId).jobList(1).show = 0
    #             self.MoList(moId).state = self.cInit
    #             self.JobCount(moId) = 1

    #         case {'Mix','Dlt'} # Mixing or Diluting
    #             # 3 jobs required
    #             for jobId=1:2:
    #                 self.MoList(moId).jobList(jobId).id = jobId
    #                 self.MoList(moId).jobList(jobId).name = ['J',int2str(jobId)]
    #                 self.MoList(moId).jobList(jobId).droplet = 
    #                     self.MoList(moId).droplets{jobId}
    #                 self.MoList(moId).jobList(jobId).goal = 
    #                     self.MoList(moId).locations{jobId}
    #                 self.MoList(moId).jobList(jobId).hazard = self.pfcnGetHazard(
    #                     self.MoList(moId).droplets{jobId},
    #                     self.MoList(moId).locations{jobId})
    #                 self.MoList(moId).jobList(jobId).kmax = self.pfcnGetKmax(
    #                     self.MoList(moId).jobList(jobId).droplet,
    #                     self.MoList(moId).jobList(jobId).goal)
    #                 self.MoList(moId).jobList(jobId).show = 0
    #             # end
    #             self.MoList(moId).state = self.cInit
    #             self.JobCount(moId) = 2

    #         case {'Mag','Was','Thm'} # Magnetic Sense
    #             self.MoList(moId).jobList(1).id = 1
    #             self.MoList(moId).jobList(1).name = 'J1'
    #             self.MoList(moId).jobList(1).droplet = self.MoList(moId).droplets{1}
    #             self.MoList(moId).jobList(1).goal = self.MoList(moId).locations{1}
    #             self.MoList(moId).jobList(1).hazard = self.pfcnGetHazard(
    #                 self.MoList(moId).droplets{1},self.MoList(moId).locations{1})
    #             self.MoList(moId).jobList(1).kmax = self.pfcnGetKmax(
    #                 self.MoList(moId).jobList(1).droplet,
    #                 self.MoList(moId).jobList(1).goal)
    #             self.MoList(moId).jobList(1).show = 0
    #             self.MoList(moId).state = self.cInit
    #             self.JobCount(moId) = 1

    #         case {'Spt'} # Split
    #             for jobId=1:2
    #                 self.MoList(moId).jobList(jobId).id = jobId
    #                 self.MoList(moId).jobList(jobId).name = ['J',int2str(jobId)]
    #                 self.MoList(moId).jobList(jobId).droplet = 
    #                     self.MoList(moId).droplets{jobId}
    #                 self.MoList(moId).jobList(jobId).goal = 
    #                     self.MoList(moId).locations{jobId}
    #                 self.MoList(moId).jobList(jobId).hazard = self.pfcnGetHazard(
    #                     self.MoList(moId).droplets{jobId},
    #                     self.MoList(moId).locations{jobId})
    #                 self.MoList(moId).jobList(jobId).kmax = self.pfcnGetKmax(
    #                     self.MoList(moId).jobList(jobId).droplet,
    #                     self.MoList(moId).jobList(jobId).goal)
    #                 self.MoList(moId).jobList(jobId).show = 0
    #             # end
    #             self.MoList(moId).state = self.cInit
    #             self.JobCount(moId) = 2

    #         otherwise:
    #             fprintf('Unknown type #s!\n',self.MoList(moId).type)
    #             return
    #     # end
    #     results = True
    #     return results
    # # end
    
    def createRoutingJobs(self) -> None:
    """ Create routing jobs for the current MO """
    # [TODO] Implement Mo.createRoutingJobs()
    if self.s_type in {'Dis'}: # Dispense
        # No job required
        rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0])
        self.rj_list.append(rjObj)
        
    elif self.s_type in {'Out','Dsc'}:
        rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0])
        self.rj_list.append(rjObj)

    elif self.s_type in {'Mix','Dlt'}: # Mixing or Diluting
        rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0])
        self.rj_list.append(rjObj)
        rjObj:Rj = Rj('J1', self.droplets[1], self.locations[0])
        self.rj_list.append(rjObj)
        
        # 3 jobs required
        for jobId in range(2):
            self.jobList[jobId].id = jobId
            self.jobList[jobId].name = ['J',int2str(jobId)]
            self.jobList[jobId].droplet = self.droplets[jobId]
            self.jobList[jobId].goal = self.locations[jobId]
            self.jobList[jobId].hazard = self.pfcnGetHazard(self.droplets[jobId], self.locations[jobId])
            self.jobList[jobId].kmax = self.pfcnGetKmax(self.jobList[jobId].droplet,self.jobList[jobId].goal)
            self.jobList[jobId].show = 0
        # end
        self.state = self.cInit
        self.JobCount(moId) = 2
    elif self.s_type in {'Mag','Was','Thm'}:
        self.jobList(1).id = 1
        self.jobList(1).name = 'J1'
        self.jobList(1).droplet = self.droplets[1]
        self.jobList(1).goal = self.locations[1]
        self.jobList(1).hazard = self.pfcnGetHazard(
            self.droplets[1],self.locations[1])
        self.jobList(1).kmax = self.pfcnGetKmax(
            self.jobList(1).droplet,
            self.jobList(1).goal)
        self.jobList(1).show = 0
        self.state = self.cInit
        self.JobCount(moId) = 1

    elif self.s_type in {'Spt'}: # Split
        for jobId in range(2):
            self.jobList[jobId].id = jobId
            self.jobList[jobId].name = ['J',int2str(jobId)]
            self.jobList[jobId].droplet = self.droplets[jobId]
            self.jobList[jobId].goal = self.locations[jobId]
            self.jobList[jobId].hazard = self.pfcnGetHazard( self.droplets[jobId], self.locations[jobId])
            self.jobList[jobId].kmax = self.pfcnGetKmax( self.jobList[jobId].droplet, self.jobList[jobId].goal)
            self.jobList[jobId].show = 0
        # end
        self.state = self.cInit
        self.JobCount[moId] = 2

    else:
        print('Unknown type %s!\n' % self.s_type)

    return

    # FUNCTION Process All Jobs
    def results = mdProcessAllJobs(self,bReSynth,bGenModel,bDStep,bLoadTb,bShortest)
        for moObj in self.MoList:
            for jobObj in moObj.jobList:
                self.mdSynthesizeJobStr(moId,jobId,bReSynth,bGenModel,bDStep,bLoadTb,bShortest)
            # end
            self.MoList(moId).state = self.cReady
        # end
        results = True
    # end

    # FUNCTION Process Job
    def mdSynthesizeJobStr(self,moId,jobId,bReSynth,bGenModel,bDStep,bLoadTb,bShortest):
        #TODO
        tic

        switch(self.MoList(moId).type)
            case 'Dis'
                self.MoList(moId).jobList(jobId).pMaxVal = 1
            otherwise
                # Health matrix
                if(bShortest):
                    tmpHealthMatrix = (2^self.Bits-1)*ones(size(self.HealthMatrix))
                else:
                    tmpHealthMatrix = self.HealthMatrix
                # end
                # Current job
                tmpJob = self.MoList(moId).jobList(jobId)
                # Check if strategy already exists
                [bExist,UniqueID] = fcnLookupStrategy(
                    self.sPostFix,
                    tmpJob.droplet,tmpJob.goal,tmpJob.hazard,tmpHealthMatrix)
                # If no strategy exists and resynth required
                if ((bExist==0 && bReSynth) || self.bForceResynth):
                    VarList = {
                        'cInitXa',int2str(tmpJob.droplet(1))
                        'cInitYa',int2str(tmpJob.droplet(2))
                        'cInitXb',int2str(tmpJob.droplet(3))
                        'cInitYb',int2str(tmpJob.droplet(4))
                        'cHazXa' ,int2str(tmpJob.hazard(1))
                        'cHazYa' ,int2str(tmpJob.hazard(2))
                        'cHazXb' ,int2str(tmpJob.hazard(3))
                        'cHazYb' ,int2str(tmpJob.hazard(4))
                        'cGoalXa',int2str(tmpJob.goal(1))
                        'cGoalYa',int2str(tmpJob.goal(2))
                        'cGoalXb',int2str(tmpJob.goal(3))
                        'cGoalYb',int2str(tmpJob.goal(4))
                        'Kmax',   int2str(tmpJob.kmax)}
                    # Check if double step model required
                    if(bDStep):
                        tmpModelNM = 'MEDAX'
                    else:
                        tmpModelNM = 'MEDAY'
                    # end
                    # Add model type
                    tmpModelNM = [tmpModelNM,'_',self.ModelType]
                    tmpPropsNM = ['MEDA','_',self.ModelType]
                    # Check if model generation is required
                    if(bGenModel):
                        fcnGenerateModel(self.ModelType,0,
                            self.Width,self.Height,
                            tmpJob.hazard,tmpHealthMatrix,self.Bits,bDStep)
                    # end
                    [~,~] = self.pfcnRunPrism(tmpModelNM,tmpPropsNM,
                        self.MoList(moId).name,
                        self.MoList(moId).jobList(jobId).name,
                        VarList,
                        UniqueID)
                    #disp(tmpY)
                # end
                # Parse output files
                #UniqueID = ''
                [self.MoList(moId).jobList(jobId).strTB,
                    self.MoList(moId).jobList(jobId).pMaxVal] = 
                    self.pfcnParseStrFile(moId,jobId,bLoadTb,UniqueID)
        # end
        tmpTime = toc
        fprintf('\t\t#s.#s #s, pmax=#1.5f (#1.3fs)\n',
            self.MoList(moId).name,
            self.MoList(moId).jobList(jobId).name,
            self.MoList(moId).type,
            self.MoList(moId).jobList(jobId).pMaxVal,
            tmpTime)
    # end

    # FUNCTION Get Hazard Zone
    def hazard = pfcnGetHazard(self,vDroplet,vLocation)
        #tmpA = droplets'
        #tmpB = locations'
        tmpAll = [vDroplet;vLocation]
        hazard = [min(tmpAll(:,1:2))-2,max(tmpAll(:,3:4))+2]
    # end

    # FUNCTION Parse Strategy File
    def [StrTable,pMaxVal] = pfcnParseStrFile(self,moId,jobId,bLoadTb,UniqueID)
        # Files
        uniqueName = [self.OutputPath,'\',
            UniqueID]
        strFileName = [uniqueName,'_','adv.txt']
        staFileName = [uniqueName,'_','sta.txt']
        resFileName = [uniqueName]
        if (bLoadTb):
            if(strcmp(self.ModelType,'SMG')):
                # Strategy File
                StrTB = readtable(strFileName,
                    'FileType','text',
                    'Delimiter',{',',':'})
                StrTB.Var4 = []
                StrTB.Properties.VariableNames = {'vSID','vK','vActName'}
                # State File
                StaTB = readtable(staFileName,
                    'FileType','text',
                    'HeaderLines', 1,
                    'ConsecutiveDelimitersRule', 'join',
                    'Delimiter',{':','[',']',','})
                StaMat = table2array(StaTB)
                # Cross Reference
                vState = nan(size(StrTB,1),5)
                for i = 1:size(StrTB,1)
                    tmpRow = StaMat(:,1)==StrTB.vSID(i)
                    vState(i,:) = StaMat(tmpRow,2:6)
                # end
                StrTB = [table(vState),StrTB]
                # Return results
                StrTable = StrTB
                # Note: to obtain value from StrTB:
                # find(ismember(StrTB.vState,x,'rows') & StrTB.vK==6 )
            elif (strcmp(self.ModelType,'MDP')):
                #TODO
                StrTB = readtable(strFileName,
                    'FileType','text',
                    'Delimiter',{' '})
                if(size(StrTB,2)==4):
                    StrTB.Var2 = []
                    StrTB.Var3 = []
                    StrTB.Properties.VariableNames = {'vSID','vActName'}
                    # State File
                    StaTB = readtable(staFileName,
                        'FileType','text',
                        'HeaderLines', 1,
                        'ConsecutiveDelimitersRule', 'join',
                        'Delimiter',{':','[',']',','})
                    StaMat = table2array(StaTB)
                    # Cross Reference
                    vState = nan(size(StrTB,1),5)
                    for i = 1:size(StrTB,1)
                        tmpRow = StaMat(:,1)==StrTB.vSID(i)
                        vState(i,:) = StaMat(tmpRow,2:6)
                    # end
                    StrTB = [table(vState),StrTB]
                    # Return results
                    StrTable = StrTB
                    # Note: to obtain value from StrTB:
                    # find(ismember(StrTB.vState,x,'rows') & StrTB.vK==6 )
                else:
                    fprintf('Warning: #s.#s no strategy found\n',
                        self.MoList(moId).name,
                        self.MoList(moId).jobList(jobId).name)
                    StrTable = []
                # end
            else:
                StrTable = []
            # end
            # Results file
            fID = fopen(resFileName,'r')
            ResCells = textscan(fID,'#s')
            pMaxVal = str2double(ResCells{1}{2})
            fclose(fID)
        else:
            StrTable = []
            pMaxVal = NaN
        # end
    # end

    # FUNCTION Get action from job
    def Action = pfcnGetAction(self,moId,jobId,vState,vK)
        if (~isempty(self.MoList(moId).jobList(jobId).strTB)):
            switch self.ModelType
                case 'SMG'
                    vK = min(vK,self.MoList(moId).jobList(jobId).kmax)
                    rowIdx = 
                        all(self.MoList(moId).jobList(jobId).strTB.vState==vState,2) & 
                        self.MoList(moId).jobList(jobId).strTB.vK==vK 
                    actionName = self.MoList(moId).jobList(jobId).strTB.vActName(rowIdx)
                    if isempty(actionName):
                        fprintf('Warning: #s.#s could not find state #s\n',
                            self.MoList(moId).name,
                            self.MoList(moId).jobList(jobId).name,
                            int2str(vState))
                        actionName = ''
                    else:
                        actionName = actionName{:}
                    # end
                    actionData = []
                case 'MDP'
                    #TODO
                    rowIdx = 
                        all(self.MoList(moId).jobList(jobId).strTB.vState==vState,2)
                    actionName = self.MoList(moId).jobList(jobId).strTB.vActName(rowIdx,1)
                    if ( isempty(actionName) &&
                            all(self.MoList(moId).jobList(jobId).goal==vState(2:5)) ):
                        actionName = 'aH'
                    elif ( isempty(actionName) ):
                        fprintf('Warning: #s.#s could not find state #s\n',
                            self.MoList(moId).name,
                            self.MoList(moId).jobList(jobId).name,
                            int2str(vState))
                        actionName = ''
                    else:
                        actionName = actionName{1}
                    # end
                    actionData = []
            # end
        elif (strcmp(self.MoList(moId).type,'Dis')):
            actionName = 'aDispense'
            tmpDelta = self.MoList(moId).jobList(jobId).goal - vState(2:5)
            tmpDelta = sign(tmpDelta).*min(abs(tmpDelta),[2 2 2 2])
            actionData = vState(2:5)+tmpDelta
        else:
            #fprintf('Warning: no strTB found for #s.#s\n',
            #    self.MoList(moId).name, self.MoList(moId).jobList(jobId).name)
            actionName = 'aH'
            actionData = []
        # end
        Action = {actionName,actionData}
    # end

    # FUNCTION Run Prism
    def [tmpX,tmpY] = pfcnRunPrism(
            self,vModelFN,vPropFN,vMoName,vJobName,vVars,vUniqueID)
        # Store current path
        tmpOriginalPath = pwd
        cd(self.PrismBinPath)
        # Objective ID
        objId = int2str(1); # q index starts at 1
        # Create script name
        uniqueName = [vUniqueID]
        scriptName = [ vModelFN,'_',vMoName,'_',vJobName,'.bat' ]
        #fprintf('#s\t Creating ',scriptName)
        # Create script contents
        fID = fopen(scriptName,'w')
        fprintf(fID,'SET MODEL_FILE=#s.prism\n',vModelFN)
        fprintf(fID,'SET PROPS_FILE=#s.props\n',vPropFN)
        fprintf(fID,'SET ADV_FILE=#s\n',uniqueName)
        fprintf(fID,'SET PVEC_FILE=#s#s.m\n',uniqueName,'_pvec')
        fprintf(fID,'SET PROP_ID=#s\n',objId)
        fprintf(fID,'#s#s\n','SET PROJ_DIR=',self.ProjectPath)
        fprintf(fID,'#s#s\n','SET OUT_DIR=',self.OutputPath)
        tmpStr = ''
        for i = 1:length(vVars):
            if (~isempty(tmpStr)):
                tmpStr = [tmpStr,',']
            # end
            tmpStr = [tmpStr, sprintf('#s=#s',vVars{i,1},vVars{i,2})]
        # end
        fprintf(fID, 'SET OP_CONST=#s\n', tmpStr)
        fprintf(fID,'#s\n', [ 
            'prism ' 
            '#PROJ_DIR#\#MODEL_FILE# ' 
            '#PROJ_DIR#\#PROPS_FILE# -explicit ' 
            '-const #OP_CONST# ' 
            '-prop #PROP_ID# ' 
            '-exportstrat #OUT_DIR#\#ADV_FILE#_adv.txt:type=actions ' 
            '-exportstates #OUT_DIR#\#ADV_FILE#_sta.txt ' 
            '-exportresults #OUT_DIR#\#ADV_FILE# ' 
            #'-v' 
            ])
        #'-exportstates #OUT_DIR#\#ADV_FILE#_sta.txt ' 
        fclose(fID)
        # Run script
        fprintf('\t ')
        [tmpX, tmpY] = system(scriptName)
        fprintf('Code #d\t',tmpX)
        if (tmpX ~=0):
            tmpY
            cd(tmpOriginalPath)
            ME = MException('Synth:PrismError', 
                'Script #s failed -- are you in the right directory?', 
                scriptName)
            throw(ME)
        # end
        # Create log file
        fID = fopen(['logs\',scriptName,'.log'],'w')
        fprintf(fID,'#s',tmpY)
        fclose(fID)
        cd(tmpOriginalPath)
    # end

    # FUNCTION Estimate kmax
    def kMax = pfcnGetKmax(self,dr1,dr2)
        tmpDist = fcnDrGetDistance(dr1,dr2)
        kMax = ceil(tmpDist*1.5)+1
    # end
# end












# end
