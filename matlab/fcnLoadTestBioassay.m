function fcnLoadTestBioassay(ScObj,moID,drSize,bcSize,iCount)
    
%     x1j1s = [2, 2, 2+drSize(1)-1, 2+drSize(2)-1];
%     x1j1g = [bcSize(1)-2-drSize(1)+1, bcSize(2)-2-drSize(2)+1, bcSize(1)-2, bcSize(2)-2];
    for iDr = 1:size(drSize,1)
        for iBc = 1:size(bcSize,1)
            for iCo = 1:iCount
                x1j1s = [2, 2, 2+drSize(iDr,1)-1, 2+drSize(iDr,2)-1];
                x1j1g = [bcSize(iBc,1)-2-drSize(iDr,1)+1, bcSize(iBc,2)-2-drSize(iDr,2)+1,...
                    bcSize(iBc,1)-2, bcSize(iBc,2)-2];
                fprintf('%d %d %d: %d - %d ', iDr, iBc, iCo, x1j1s, x1j1g);
                fprintf('\n');
                moName = sprintf('x%01d%01d_%02d', iDr, iBc, iCo);
                ScObj.mdAddMo(moName,'Mag',{},{x1j1s},{x1j1g});
            end
        end
    end

end