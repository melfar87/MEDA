function TB2 = fcnLoadCP2Bioassay(ScObj)

TB = {...
    {'a01'}, {'Dis'}, {}, {}, {[17.5, 2.5]};...
    {'a02'}, {'Dis'}, {}, {}, {[47.5, 2.5]};...
    {'a03'}, {'Dis'}, {}, {}, {[58.5, 24.5]};...
    {'a04'}, {'Dis'}, {}, {}, {[58.5, 9.5]};...
    {'a05'}, {'Mix'}, {'a01','a02'}, {}, {[34.5, 6.5]};...
    {'a06'}, {'Mix'}, {'a03','a04'}, {}, {[56.5, 18.5]};...
    {'a07'}, {'Mix'}, {'a05','a06'}, {}, {[52.5, 6.5]};...
    {'b02'}, {'Dis'}, {}, {'a07'}, {[17.5, 28.5]};...
    {'b03'}, {'Mix'}, {'a07','b02'}, {}, {[18.5, 6.5]};...
    {'b04'}, {'Mag'}, {'b03'}, {}, {[23.5, 6.5]};...
    {'b05'}, {'Spt'}, {'b04'}, {}, {[23.5, 6.5]};...
    {'c02'}, {'Dis'}, {}, {'b03'}, {[47.5, 2.5]};...
    {'c03'}, {'Mix'}, {'b05','c02'}, {}, {[34.5, 6.5]};...
    {'c04'}, {'Dis'}, {}, {'c03'}, {[47.5, 28.5]};...
    {'c05'}, {'Dlt'}, {'c03','c04'}, {}, {[49.5, 10.5]};...
    };




TB = cell2table(TB);
TB.Properties.VariableNames = {'vName','vType','vPre','vCond','vCp'};
vCp = TB.vCp;




%%
%return

% Type
vSize = cell(size(TB,1),1);
vDr   = cell(size(TB,1),1);
vDroplets = cell(size(TB,1),1);
vLocations = cell(size(TB,1),1);

% Compute initial sizes
for r = 1:size(TB,1)
    if (strcmp(TB.vType{r},'Dis'))
        vSize{r} = 16;
    else
        vSize{r} = 0;
    end
end

%% Process Information
% vName
% vType
% vPre
% vDroplets
% vLocations

% Given: {'vName','vType','vPre','vCond','vCp'}



%% Compute sizes

bRepeat = true;
while (bRepeat)
    bRepeat = false;
    for r = 1:size(TB,1)
        if (vSize{r}==0)
            if (isempty(TB.vPre{r}))
                vSize{r} = fcnDrGetSize(vDr{r});
            else
                tmpIsReady = true;
                for i = 1:length(TB.vPre{r})
                    tmpIsReady = tmpIsReady && vSize{strcmp(TB.vName,TB.vPre{r}{i})};
                end
                if (tmpIsReady)
                    idx1 = strcmp(TB.vName,TB.vPre{r}{1});
                    if (length(TB.vPre{r})==2)
                        idx2 = strcmp(TB.vName,TB.vPre{r}{2});
                    else
                        idx2 = 0;
                    end
                    switch TB.vType{r}
                        case 'Mix'
                            vSize{r} = vSize{idx1}+vSize{idx2};
                        case 'Dlt'
                            vSize{r} = (vSize{idx1}+vSize{idx2})/2;
                        case 'Spt'
                            vSize{r} = vSize{idx1}/2;
                        case 'Mag'
                            vSize{r} = vSize{idx1};
                        case 'Out'
                            vSize{r} = vSize{idx1};
                        otherwise
                            fprintf('Error: Unknown type %s\n', TB.vType{r});
                            return
                    end
                else
                    bRepeat = true;
                end
            end
        end
    end
end

%% Compute Droplets
for r = 1:size(TB,1)
    vDr{r} = fcnDrGetDroplet(vCp{r},vSize{r});
end

%% Compute droplets and locations

for r = 1:size(TB,1)
    if (length(TB.vPre{r})==1)
        idx1 = strcmp(TB.vName,TB.vPre{r}{1});
        idx2 = 0;
    elseif (length(TB.vPre{r})==2)
        idx1 = strcmp(TB.vName,TB.vPre{r}{1});
        idx2 = strcmp(TB.vName,TB.vPre{r}{2});
    else
        idx1 = 0;
        idx2 = 0;
    end
    switch TB.vType{r}
        case 'Dis'
            vDroplets{r} = {fcnDrGetDisDroplet(vDr{r},60,30)};
            vLocations{r} = vDr(r);
        case 'Mix'
            vDroplets{r} = {...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1}),...
                fcnDrGetDroplet(vCp{idx2},vSize{idx2})};
            vLocations{r} = {...
                fcnDrGetDroplet(vCp{r},vSize{idx1}),...
                fcnDrGetDroplet(vCp{r},vSize{idx2})};
        case 'Dlt'
            vDroplets{r} = {...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1}),...
                fcnDrGetDroplet(vCp{idx2},vSize{idx2})};
            vLocations{r} = {...
                fcnDrGetDroplet(vCp{r},vSize{idx1}),...
                fcnDrGetDroplet(vCp{r},vSize{idx2})};
        case 'Spt'
            vDroplets{r} = {...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1}/2),...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1}/2)};
            vLocations{r} = {...
                fcnDrGetDroplet(vCp{r}-[4 0],vSize{idx1}/2),...
                fcnDrGetDroplet(vCp{r}+[4 0],vSize{idx1}/2)};
        case 'Mag'
            vDroplets{r} = {...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1})};
            vLocations{r} = {...
                fcnDrGetDroplet(vCp{r},vSize{idx1})};
        case 'Out'
            vDroplets{r} = {...
                fcnDrGetDroplet(vCp{idx1},vSize{idx1})};
            vLocations{r} = {...
                fcnDrGetDroplet(vCp{r},vSize{idx1})};
        otherwise
            fprintf('Error: Unknown type %s', vType{r});
            return
    end
end


%return

TB2 = [TB,...
    cell2table(vDroplets),...
    cell2table(vLocations),...
    cell2table(vSize),...
    cell2table(vDr)];

% Correct splits
splitCount = zeros(size(TB2,1),1);
for r = 1:size(TB2,1)
    preIdx = {[],[]};
    for preLocIdx = 1:size(TB2.vPre{r})
        preIdx{preLocIdx} = strcmp(TB2.vName,TB2.vPre{r}{preLocIdx});
        if (strcmp(TB2.vType{preIdx{preLocIdx}},'Spt'))
            if (splitCount(preIdx{preLocIdx})==0)
                TB2.vDroplets{r}{preLocIdx} = TB2.vDroplets{r}{preLocIdx} - [4 0 4 0];
                splitCount(preIdx{preLocIdx}) = splitCount(preIdx{preLocIdx}) + 1;
            else
                TB2.vDroplets{r}{preLocIdx} = TB2.vDroplets{r}{preLocIdx} + [4 0 4 0];
            end
        end
    end
end
    %if (strcmp(TB2.vType{r},'Spt')

for r = 1:size(TB2,1)
    %ScObj.mdAddMo('a5','Mix',{'a1','a2'},{a5j1s,a5j2s},{a5j1g,a5j2g});
    ScObj.mdAddMo(...
        TB2.vName{r},...
        TB2.vType{r},...
        TB2.vPre{r},...
        TB2.vDroplets{r},...
        TB2.vLocations{r},...
        TB2.vCond{r});
end

% Sanity checks:
for moId = 1:length(ScObj.MoList)
    for jobId = 1:length(ScObj.MoList(moId).jobList)
        % Check if start and end sizes are the same
        if ( (fcnDrGetSize(ScObj.MoList(moId).jobList(jobId).droplet))~=...
                (fcnDrGetSize(ScObj.MoList(moId).jobList(jobId).goal)) )
            fprintf('Error: %02d.%02d size mismatch!\n',moId,jobId);
            pause;
        end
        % Check if droplets are out of bounds
        if ( (~strcmp(ScObj.MoList(moId).type,'Dis')) && ...
                ( (fcnDrIsOutOfBounds(ScObj.MoList(moId).jobList(jobId).droplet)) || ...
                  (fcnDrIsOutOfBounds(ScObj.MoList(moId).jobList(jobId).goal)) ) )
            fprintf('Error: %02d.%02d out of bounds!\n',moId,jobId);
            pause;
        end
    end
end


end