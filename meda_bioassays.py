# from enum import IntEnum
# from meda_scheduler import MedaScheduler
# from typing import List
# import numpy as np
# from meda_sgs import Sg, Sgid
    

# class Bioassays():
#     """ Bioassay sequence graph class """
        
#     def fcnDrGetSize(self,dr):
#         dr_size = (dr[2]-dr[0]+1)*(dr[3]-dr[1]+1)
#         return dr_size
    
    
#     def importBioassay(self, sg: List, sch: MedaScheduler):
        
#         mo_entry:List
#         for mo_entry in sg:
#             sch.addMo(
#                 s_name=mo_entry[Sgid.NAME],
#                 s_type=mo_entry[Sgid.TYPE],
#                 s_pre=mo_entry[Sgid.PRE],
#                 dr_start=None,
#                 dr_end=mo_entry[Sgid.LOCS]
#             )
        # 
        # 
        # vSize = np.zeros(len(sg))
        # vDr = np.zeros(len(sg))
        
        # Compute initial sizes
        # for idx, mo_item in enumerate(sg):
        #     if mo_item[Sgid.TYPE][0] == 'Dis':
        #         vSize[idx] = 16
                
        # bRepeat = True
        # while (bRepeat):
        #     bRepeat = False
        #     for r, mo_item in enumerate(sg):
        #         if (vSize[r]==0):
        #             if not mo_item[Sgid.PRE]:
        #                 vSize[r] = self.fcnDrGetSize(vDr[r])
        #             else
        #                 tmpIsReady = True
        #                 for i = 1:length(TB.vPre[r])
        #                     tmpIsReady = tmpIsReady and (vSize{strcmp(TB.vName,TB.vPre[r]{i})}
        #                 end
        #                 if (tmpIsReady)
        #                     idx1 = strcmp(TB.vName,TB.vPre[r]{1})
        #                     if (length(TB.vPre[r])==2)
        #                         idx2 = strcmp(TB.vName,TB.vPre[r]{2})
        #                     else
        #                         idx2 = 0
        #                     end
        #                     switch TB.vType[r]
        #                         case 'Mix'
        #                             vSize[r] = vSize{idx1}+vSize{idx2}
        #                         case 'Dlt'
        #                             vSize[r] = (vSize{idx1}+vSize{idx2})/2
        #                         case 'Spt'
        #                             vSize[r] = vSize{idx1}/2
        #                         case {'Mag','Thm'}
        #                             vSize[r] = vSize{idx1}
        #                         case {'Out','Dsc'}
        #                             vSize[r] = vSize{idx1}
        #                         otherwise
        #                             fprintf('Error: Unknown type %s\n', TB.vType[r])
        #                             return
        #                     end
        #                 else
        #                     bRepeat = True
        #                 end
        #             end
        #         end
        #     end
        # end
                
                
        
