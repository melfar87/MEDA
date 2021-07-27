from enum import IntEnum
from typing import List
from meda_scheduler import MedaScheduler


class Sgid(IntEnum):
    NAME = 0
    TYPE = 1
    PRE  = 2
    COND = 3
    DRS  = 4
    LOCS = 5


class Bioassays():
    """ Bioassay sequence graph class, processed using MATLAB """
    def __init__(self) -> None:
        pass
    
    def importBioassay(self, sg: List, sch: MedaScheduler):
        
        mo_entry:List
        for mo_entry in sg:
            dr_list = mo_entry[Sgid.DRS] + mo_entry[Sgid.LOCS]
            for dr in dr_list:
                dr[0], dr[1] = dr[0]-1, dr[1]-1 # start from 0
            sch.addMo(
                s_name=mo_entry[Sgid.NAME],
                s_type=mo_entry[Sgid.TYPE],
                s_pre=mo_entry[Sgid.PRE],
                dr_start=mo_entry[Sgid.DRS],
                dr_end=mo_entry[Sgid.LOCS],
                cond=mo_entry[Sgid.COND]
            )
        return
    
    sg_Simple = [
        ['n00','Dis',[],[],[[8,-3,11,0]],[[8,1,11,4]]],
        ['n01','Dis',[],[],[[21,-3,24,0]],[[21,1,24,4]]],
        ['n08','Mix',['n00','n01'],[],[[8,1,11,4],[21,1,24,4]],[[14,19,17,22],[14,19,17,22]]],
    ]
    
    sg_CRAT = [
        ['n00','Dis',[],[],[[8,-3,11,0]],[[8,1,11,4]]],
        ['n01','Dis',[],[],[[48,-3,51,0]],[[48,1,51,4]]],
        ['n02','Dis',[],[],[[14,31,17,34]],[[14,26,17,29]]],
        ['n03','Dis',[],[],[[44,31,47,34]],[[44,26,47,29]]],
        ['n04','Dis',[],[],[[28,-3,31,0]],[[28,1,31,4]]],
        ['n05','Dis',[],['n08'],[[48,-3,51,0]],[[48,1,51,4]]],
        ['n06','Dis',[],['n09'],[[14,31,17,34]],[[14,26,17,29]]],
        ['n07','Dis',[],['n10'],[[44,31,47,34]],[[44,26,47,29]]],
        ['n08','Mix',['n00','n01'],[],[[8,1,11,4],[48,1,51,4]],[[14,19,17,22],[14,19,17,22]]],
        ['n09','Mix',['n02','n14'],['n14'],[[14,26,17,29],[14,9,17,12]],[[14,19,17,22],[14,19,17,22]]],
        ['n10','Mix',['n03','n15'],['n15'],[[44,26,47,29],[14,9,17,12]],[[14,19,17,22],[14,19,17,22]]],
        ['n11','Mix',['n04','n05'],[],[[28,1,31,4],[48,1,51,4]],[[44,9,47,12],[44,9,47,12]]],
        ['n12','Mix',['n06','n17'],['n17'],[[14,26,17,29],[44,19,47,22]],[[44,9,47,12],[44,9,47,12]]],
        ['n13','Mix',['n07','n18'],['n18'],[[44,26,47,29],[44,19,47,22]],[[44,9,47,12],[44,9,47,12]]],
        ['n14','Spt',['n08'],[],[[14,19,17,22],[14,19,17,22]],[[10,9,13,12],[18,9,21,12]]],
        ['n15','Spt',['n09'],['n20','n09'],[[14,19,17,22],[14,19,17,22]],[[10,9,13,12],[18,9,21,12]]],
        ['n16','Spt',['n10'],['n21','n10'],[[14,19,17,22],[14,19,17,22]],[[10,9,13,12],[18,9,21,12]]],
        ['n17','Spt',['n11'],[],[[44,9,47,12],[44,9,47,12]],[[40,19,43,22],[48,19,51,22]]],
        ['n18','Spt',['n12'],['n24','n12'],[[44,9,47,12],[44,9,47,12]],[[40,19,43,22],[48,19,51,22]]],
        ['n19','Spt',['n13'],['n25','n13'],[[44,9,47,12],[44,9,47,12]],[[40,19,43,22],[48,19,51,22]]],
        ['n20','Dsc',['n14'],[],[[10,9,13,12]],[[1,14,4,17]]],
        ['n21','Dsc',['n15'],[],[[10,9,13,12]],[[1,14,4,17]]],
        ['n22','Dsc',['n16'],[],[[10,9,13,12]],[[1,14,4,17]]],
        ['n23','Out',['n16'],[],[[10,9,13,12]],[[55,14,58,17]]],
        ['n24','Dsc',['n17'],[],[[40,19,43,22]],[[1,14,4,17]]],
        ['n25','Dsc',['n18'],[],[[40,19,43,22]],[[1,14,4,17]]],
        ['n26','Dsc',['n19'],[],[[40,19,43,22]],[[1,14,4,17]]],
        ['n27','Out',['n19'],[],[[40,19,43,22]],[[55,14,58,17]]],
    ]
    
    sg_CPCR = [
        ['n00','Dis',[],[],[15.5,2.5],[[14,-3,17,0]]],
        ['n01','Dis',[],[],[15.5,2.5],[[44,-3,47,0]]],
        ['n02','Dis',[],[],[15.5,2.5],[[14,31,17,34]]],
        ['n03','Dis',[],[],[15.5,2.5],[[44,31,47,34]]],
        ['n04','Mix',['n00','n01'],[],[15.5,2.5],[[14,1,17,4],[44,1,47,4]]],
        ['n05','Mix',['n02','n03'],[],[15.5,2.5],[[14,26,17,29],[44,26,47,29]]],
        ['n06','Spt',['n04'],[],[15.5,2.5],[[14,19,17,22],[14,19,17,22]]],
        ['n07','Spt',['n05'],['n08','n09'],[15.5,2.5],[[44,9,47,12],[44,9,47,12]]],
        ['n08','Dsc',['n06'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n09','Mix',['n06','n07'],['n06'],[15.5,2.5],[[20,9,23,12],[24,9,27,12]]],
        ['n10','Dsc',['n07'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n11','Spt',['n09'],['n10','n09'],[15.5,2.5],[[14,19,17,22],[14,19,17,22]]],
        ['n12','Dsc',['n11'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n13','Thm',['n11'],[],[15.5,2.5],[[28,9,31,12]]],
        ['n14','Thm',['n13'],[],[15.5,2.5],[[14,9,17,12]]],
        ['n15','Thm',['n14'],['n14'],[15.5,2.5],[[44,19,47,22]]],
        ['n16','Thm',['n15'],['n15'],[15.5,2.5],[[14,9,17,12]]],
        ['n17','Thm',['n16'],['n16'],[15.5,2.5],[[44,19,47,22]]],
        ['n18','Thm',['n17'],['n17'],[15.5,2.5],[[14,9,17,12]]],
        ['n19','Thm',['n18'],['n18'],[15.5,2.5],[[44,19,47,22]]],
        ['n20','Thm',['n19'],['n19'],[15.5,2.5],[[14,9,17,12]]],
        ['n21','Thm',['n20'],['n20'],[15.5,2.5],[[44,19,47,22]]],
        ['n22','Thm',['n21'],['n21'],[15.5,2.5],[[14,9,17,12]]],
        ['n23','Thm',['n22'],['n22'],[15.5,2.5],[[44,19,47,22]]],
        ['n24','Thm',['n23'],['n23'],[15.5,2.5],[[14,9,17,12]]],
        ['n25','Thm',['n24'],['n24'],[15.5,2.5],[[44,19,47,22]]],
        ['n26','Thm',['n25'],['n25'],[15.5,2.5],[[14,9,17,12]]],
        ['n27','Thm',['n26'],['n26'],[15.5,2.5],[[44,19,47,22]]],
        ['n28','Thm',['n27'],['n27'],[15.5,2.5],[[14,9,17,12]]],
        ['n29','Thm',['n28'],['n28'],[15.5,2.5],[[44,19,47,22]]],
        ['n30','Thm',['n29'],['n29'],[15.5,2.5],[[14,9,17,12]]],
        ['n31','Thm',['n30'],['n30'],[15.5,2.5],[[44,19,47,22]]],
        ['n32','Thm',['n31'],['n31'],[15.5,2.5],[[14,9,17,12]]],
        ['n33','Out',['n32'],[],[15.5,2.5],[[44,19,47,22]]],
    ]
    
    sg_CP2 = [
        ['a01','Dis',[],[],[17.5,2.5],[[16,-3,19,0]]],
        ['a02','Dis',[],[],[17.5,2.5],[[46,-3,49,0]]],
        ['a03','Dis',[],[],[17.5,2.5],[[61,23,64,26]]],
        ['a04','Dis',[],[],[17.5,2.5],[[61,8,64,11]]],
        ['a05','Mix',['a01','a02'],[],[17.5,2.5],[[16,1,19,4],[46,1,49,4]]],
        ['a06','Mix',['a03','a04'],[],[17.5,2.5],[[57,23,60,26],[57,8,60,11]]],
        ['a07','Mix',['a05','a06'],[],[17.5,2.5],[[32,4,37,8],[54,16,59,20]]],
        ['b02','Dis',[],['a07'],[17.5,2.5],[[16,31,19,34]]],
        ['b03','Mix',['a07','b02'],[],[17.5,2.5],[[49,3,56,10],[16,27,19,30]]],
        ['b04','Mag',['b03'],[],[17.5,2.5],[[14,2,22,10]]],
        ['b05','Spt',['b04'],[],[17.5,2.5],[[20,4,26,9],[20,4,26,9]]],
        ['c02','Dis',[],['b03'],[17.5,2.5],[[46,-3,49,0]]],
        ['c03','Mix',['b05','c02'],[],[17.5,2.5],[[16,4,22,9],[46,1,49,4]]],
        ['c04','Dis',[],['c03'],[17.5,2.5],[[46,31,49,34]]],
        ['c05','Dlt',['c03','c04'],[],[17.5,2.5],[[31,3,38,9],[46,27,49,30]]],
    ]
    
    sg_SD2 = [
        ['n00','Dis',[],[],[15.5,2.5],[[14,-3,17,0]]],
        ['n01','Dis',[],[],[15.5,2.5],[[14,31,17,34]]],
        ['n02','Dis',[],[],[15.5,2.5],[[44,31,47,34]]],
        ['n03','Dis',[],['n16'],[15.5,2.5],[[14,-3,17,0]]],
        ['n04','Dis',[],[],[15.5,2.5],[[44,-3,47,0]]],
        ['n05','Dis',[],['n17'],[15.5,2.5],[[14,31,17,34]]],
        ['n06','Dis',[],['n18'],[15.5,2.5],[[44,31,47,34]]],
        ['n07','Dis',[],['n19'],[15.5,2.5],[[44,-3,47,0]]],
        ['n08','Dis',[],['n20'],[15.5,2.5],[[14,31,17,34]]],
        ['n09','Dis',[],['n21'],[15.5,2.5],[[44,31,47,34]]],
        ['n10','Dis',[],['n22'],[15.5,2.5],[[44,-3,47,0]]],
        ['n11','Dis',[],['n23'],[15.5,2.5],[[14,31,17,34]]],
        ['n12','Dis',[],['n24'],[15.5,2.5],[[44,31,47,34]]],
        ['n13','Dis',[],['n25'],[15.5,2.5],[[44,-3,47,0]]],
        ['n14','Dis',[],['n26'],[15.5,2.5],[[14,31,17,34]]],
        ['n15','Dis',[],['n27'],[15.5,2.5],[[44,31,47,34]]],
        ['m16','Mag',['n00'],[],[15.5,2.5],[[14,1,17,4]]],
        ['n16','Spt',['m16'],[],[15.5,2.5],[[14,19,16,21],[14,19,16,21]]],
        ['n17','Mix',['n16','n01'],[],[15.5,2.5],[[10,19,12,21],[14,27,17,30]]],
        ['n18','Mix',['n16','n02'],[],[15.5,2.5],[[18,19,20,21],[44,27,47,30]]],
        ['m19','Mix',['n03','n04'],[],[15.5,2.5],[[14,1,17,4],[44,1,47,4]]],
        ['n19','Spt',['m19'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n20','Mix',['n33','n05'],['n31'],[15.5,2.5],[[10,19,12,21],[14,27,17,30]]],
        ['n21','Mix',['n33','n06'],['n32'],[15.5,2.5],[[18,19,20,21],[44,27,47,30]]],
        ['m22','Mix',['n19','n07'],['n33'],[15.5,2.5],[[10,9,13,12],[44,1,47,4]]],
        ['n22','Spt',['m22'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n23','Mix',['n36','n08'],['n34'],[15.5,2.5],[[10,19,12,21],[14,27,17,30]]],
        ['n24','Mix',['n36','n09'],['n35'],[15.5,2.5],[[18,19,20,21],[44,27,47,30]]],
        ['m25','Mix',['n22','n10'],['n36'],[15.5,2.5],[[10,9,13,12],[44,1,47,4]]],
        ['n25','Spt',['m25'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n26','Mix',['n39','n11'],['n37'],[15.5,2.5],[[10,19,12,21],[14,27,17,30]]],
        ['n27','Mix',['n39','n12'],['n38'],[15.5,2.5],[[18,19,20,21],[44,27,47,30]]],
        ['m28','Mix',['n25','n13'],['n39'],[15.5,2.5],[[10,9,13,12],[44,1,47,4]]],
        ['n28','Spt',['m28'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n29','Mix',['n42','n14'],['n40'],[15.5,2.5],[[10,19,12,21],[14,27,17,30]]],
        ['n30','Mix',['n42','n15'],['n41'],[15.5,2.5],[[18,19,20,21],[44,27,47,30]]],
        ['n31','Out',['n17'],[],[15.5,2.5],[[43,8,47,12]]],
        ['n32','Out',['n18'],[],[15.5,2.5],[[43,18,47,22]]],
        ['m33','Mag',['n19'],['n17','n18'],[15.5,2.5],[[18,9,21,12]]],
        ['n33','Spt',['m33'],[],[15.5,2.5],[[14,19,16,21],[14,19,16,21]]],
        ['n34','Out',['n20'],[],[15.5,2.5],[[43,8,47,12]]],
        ['n35','Out',['n21'],[],[15.5,2.5],[[43,18,47,22]]],
        ['m36','Mag',['n22'],['n20','n21'],[15.5,2.5],[[18,9,21,12]]],
        ['n36','Spt',['m36'],[],[15.5,2.5],[[14,19,16,21],[14,19,16,21]]],
        ['n37','Out',['n23'],[],[15.5,2.5],[[43,8,47,12]]],
        ['n38','Out',['n24'],[],[15.5,2.5],[[43,18,47,22]]],
        ['m39','Mag',['n25'],['n23','n24'],[15.5,2.5],[[18,9,21,12]]],
        ['n39','Spt',['m39'],[],[15.5,2.5],[[14,19,16,21],[14,19,16,21]]],
        ['n40','Out',['n26'],[],[15.5,2.5],[[43,8,47,12]]],
        ['n41','Out',['n27'],[],[15.5,2.5],[[43,18,47,22]]],
        ['m42','Mag',['n28'],['n26','n27'],[15.5,2.5],[[10,9,13,12]]],
        ['n42','Spt',['m42'],[],[15.5,2.5],[[14,19,16,21],[14,19,16,21]]],
        ['n43','Out',['n29'],[],[15.5,2.5],[[43,8,47,12]]],
        ['n44','Out',['n30'],[],[15.5,2.5],[[43,18,47,22]]],
    ]
    
    sg_MM2 = [
        ['n00','Dis',[],[],[15.5,2.5],[[14,-3,17,0]]],
        ['n01','Dis',[],[],[15.5,2.5],[[44,-3,47,0]]],
        ['n02','Dis',[],[],[15.5,2.5],[[14,31,17,34]]],
        ['n03','Dis',[],[],[15.5,2.5],[[44,31,47,34]]],
        ['n04','Dis',[],['n20'],[15.5,2.5],[[14,-3,17,0]]],
        ['n05','Dis',[],['n21'],[15.5,2.5],[[44,-3,47,0]]],
        ['n06','Dis',[],['n21'],[15.5,2.5],[[14,31,17,34]]],
        ['n07','Dis',[],['n22'],[15.5,2.5],[[44,31,47,34]]],
        ['n08','Dis',[],['n23'],[15.5,2.5],[[14,-3,17,0]]],
        ['n09','Dis',[],['n24'],[15.5,2.5],[[44,-3,47,0]]],
        ['n10','Dis',[],['n24'],[15.5,2.5],[[14,31,17,34]]],
        ['n11','Dis',[],['n25'],[15.5,2.5],[[44,31,47,34]]],
        ['n12','Dis',[],['n26'],[15.5,2.5],[[14,-3,17,0]]],
        ['n13','Dis',[],['n27'],[15.5,2.5],[[44,-3,47,0]]],
        ['n14','Dis',[],['n27'],[15.5,2.5],[[14,31,17,34]]],
        ['n15','Dis',[],['n28'],[15.5,2.5],[[44,31,47,34]]],
        ['n16','Dis',[],['n29'],[15.5,2.5],[[14,-3,17,0]]],
        ['n17','Dis',[],['n30'],[15.5,2.5],[[44,-3,47,0]]],
        ['n18','Dis',[],['n30'],[15.5,2.5],[[14,31,17,34]]],
        ['n19','Dis',[],['n31'],[15.5,2.5],[[44,31,47,34]]],
        ['n20','Mix',['n00','n21'],[],[15.5,2.5],[[14,1,17,4],[14,9,17,12]]],
        ['m21','Mix',['n01','n02'],[],[15.5,2.5],[[44,1,47,4],[14,27,17,30]]],
        ['n21','Spt',['m21'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n22','Mix',['n21','n03'],[],[15.5,2.5],[[10,9,13,12],[44,27,47,30]]],
        ['n23','Mix',['n04','n24'],[],[15.5,2.5],[[14,1,17,4],[14,9,17,12]]],
        ['m24','Mix',['n05','n06'],['n20','n22'],[15.5,2.5],[[44,1,47,4],[14,27,17,30]]],
        ['n24','Spt',['m24'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n25','Mix',['n24','n07'],['n35'],[15.5,2.5],[[10,9,13,12],[44,27,47,30]]],
        ['n26','Mix',['n08','n27'],['n36'],[15.5,2.5],[[14,1,17,4],[14,9,17,12]]],
        ['m27','Mix',['n09','n10'],['n23','n25'],[15.5,2.5],[[44,1,47,4],[14,27,17,30]]],
        ['n27','Spt',['m27'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n28','Mix',['n27','n11'],['n38'],[15.5,2.5],[[10,9,13,12],[44,27,47,30]]],
        ['n29','Mix',['n12','n30'],['n38'],[15.5,2.5],[[14,1,17,4],[14,9,17,12]]],
        ['m30','Mix',['n13','n14'],['n26','n28'],[15.5,2.5],[[44,1,47,4],[14,27,17,30]]],
        ['n30','Spt',['m30'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n31','Mix',['n30','n15'],['n39'],[15.5,2.5],[[10,9,13,12],[44,27,47,30]]],
        ['n32','Mix',['n16','n33'],['n40'],[15.5,2.5],[[14,1,17,4],[14,9,17,12]]],
        ['m33','Mix',['n17','n18'],['n29','n31'],[15.5,2.5],[[44,1,47,4],[14,27,17,30]]],
        ['n33','Spt',['m33'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n34','Mix',['n33','n19'],['n41'],[15.5,2.5],[[10,9,13,12],[44,27,47,30]]],
        ['n35','Out',['n20'],[],[15.5,2.5],[[13,18,18,22]]],
        ['n36','Out',['n22'],[],[15.5,2.5],[[43,18,48,22]]],
        ['n37','Out',['n23'],[],[15.5,2.5],[[43,18,48,22]]],
        ['n38','Out',['n25'],[],[15.5,2.5],[[43,8,48,12]]],
        ['n39','Out',['n26'],[],[15.5,2.5],[[43,8,48,12]]],
        ['n40','Out',['n28'],[],[15.5,2.5],[[13,18,18,22]]],
        ['n41','Out',['n29'],[],[15.5,2.5],[[13,18,18,22]]],
        ['n42','Out',['n31'],[],[15.5,2.5],[[43,18,48,22]]],
        ['n43','Out',['n32'],[],[15.5,2.5],[[43,18,48,22]]],
        ['n44','Out',['n34'],[],[15.5,2.5],[[43,8,48,12]]],
    ]

    sg_Nuip8 = [
        ['n00','Dis',[],[],[15.5,2.5],[[14,-3,17,0]]],
        ['n01','Dis',[],[],[15.5,2.5],[[44,-3,47,0]]],
        ['n02','Dis',[],[],[15.5,2.5],[[44,31,47,34]]],
        ['n03','Dis',[],['n13'],[15.5,2.5],[[44,31,47,34]]],
        ['n04','Dis',[],['n14'],[15.5,2.5],[[44,31,47,34]]],
        ['n05','Dis',[],['n15'],[15.5,2.5],[[44,31,47,34]]],
        ['n06','Dis',[],['n16'],[15.5,2.5],[[44,31,47,34]]],
        ['n07','Dis',[],['n17'],[15.5,2.5],[[44,31,47,34]]],
        ['n08','Dis',[],['n18'],[15.5,2.5],[[44,31,47,34]]],
        ['n09','Dis',[],['n19'],[15.5,2.5],[[44,31,47,34]]],
        ['n10','Dis',[],[],[15.5,2.5],[[14,31,17,34]]],
        ['n11','Dis',[],['n20'],[15.5,2.5],[[44,31,47,34]]],
        ['n12','Mix',['n00','n01'],[],[15.5,2.5],[[14,1,17,4],[44,1,47,4]]],
        ['n13','Mix',['n32','n02'],[],[15.5,2.5],[[20,9,23,12],[44,26,47,29]]],
        ['n14','Mix',['n33','n03'],[],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n15','Mix',['n34','n04'],['n23'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n16','Mix',['n35','n05'],['n24'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n17','Mix',['n36','n06'],['n25'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n18','Mix',['n37','n07'],['n26'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n19','Mix',['n38','n08'],['n27'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n20','Mix',['n39','n09'],['n28'],[15.5,2.5],[[28,9,31,12],[44,26,47,29]]],
        ['n21','Mix',['n20','n10'],['n29'],[15.5,2.5],[[43,18,48,22],[14,26,17,29]]],
        ['n22','Mix',['n21','n11'],['n30'],[15.5,2.5],[[12,17,18,23],[44,26,47,29]]],
        ['n23','Mag',['n12'],[],[15.5,2.5],[[13,18,18,22]]],
        ['n24','Mag',['n13'],['n32'],[15.5,2.5],[[43,8,48,12]]],
        ['n25','Mag',['n14'],['n33'],[15.5,2.5],[[43,18,48,22]]],
        ['n26','Mag',['n15'],['n34'],[15.5,2.5],[[13,18,18,22]]],
        ['n27','Mag',['n16'],['n35'],[15.5,2.5],[[43,8,48,12]]],
        ['n28','Mag',['n17'],['n36'],[15.5,2.5],[[43,18,48,22]]],
        ['n29','Mag',['n18'],['n37'],[15.5,2.5],[[13,18,18,22]]],
        ['n30','Mag',['n19'],['n38'],[15.5,2.5],[[43,8,48,12]]],
        ['n31','Mag',['n22'],['n39'],[15.5,2.5],[[42,7,49,14]]],
        ['n32','Spt',['n23'],[],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n33','Spt',['n24'],['n13','n41'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n34','Spt',['n25'],['n14','n42'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n35','Spt',['n26'],['n15','n43'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n36','Spt',['n27'],['n16','n44'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n37','Spt',['n28'],['n17','n45'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n38','Spt',['n29'],['n18','n46'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n39','Spt',['n30'],['n19','n47'],[15.5,2.5],[[14,9,17,12],[14,9,17,12]]],
        ['n40','Spt',['n31'],['n20','n48'],[15.5,2.5],[[13,8,18,12],[13,8,18,12]]],
        ['n41','Dsc',['n32'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n42','Dsc',['n33'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n43','Dsc',['n34'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n44','Dsc',['n35'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n45','Dsc',['n36'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n46','Dsc',['n37'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n47','Dsc',['n38'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n48','Dsc',['n39'],[],[15.5,2.5],[[20,9,23,12]]],
        ['n49','Out',['n40'],[],[15.5,2.5],[[27,8,32,12]]],
        ['n50','Dsc',['n40'],[],[15.5,2.5],[[19,8,24,12]]],
    ]


class SgRaw():
    """ Bioassay sequence graph class """
    
    # Default size for dispensed droplets
    dispense_size = [4,4]
    
    sg_simple = [
        [['n00'], ['Dis'], [], [], [6.5,2.5], []],
        [['n00'], ['Mag'], [], [], [15.5,20.5], []]
    ]
    
    sg_covid_rat = [
        [['n00'], ['Dis'], [], [], [10.0, 2.5], []],
        [['n01'], ['Dis'], [], [], [50.0, 2.5], [1, 1]],
        [['n02'], ['Dis'], [], [], [15.5, 27.5], [1, 1]],
        [['n03'], ['Dis'], [], [], [45.5, 27.5], [1, 1]],
        [['n04'], ['Dis'], [], [], [30.0, 2.5], [1, 1]],
        [['n05'], ['Dis'], [], ['n08'], [50.0, 2.5], [1, 1]],
        [['n06'], ['Dis'], [], ['n09'], [15.5, 27.5], [1, 1]],
        [['n07'], ['Dis'], [], ['n10'], [45.5, 27.5], [1, 1]],
        [['n08'], ['Mix'], ['n00', 'n01'], [], [15.5, 20.5], [1, 1]],
        [['n09'], ['Mix'], ['n02', 'n14'], ['n14'], [15.5, 20.5], [1, 1]],
        [['n10'], ['Mix'], ['n03', 'n15'], ['n15'], [15.5, 20.5], [1, 1]],
        [['n11'], ['Mix'], ['n04', 'n05'], [], [45.5, 10.5], [1, 1]],
        [['n12'], ['Mix'], ['n06', 'n17'], ['n17'], [45.5, 10.5], [1, 1]],
        [['n13'], ['Mix'], ['n07', 'n18'], ['n18'], [45.5, 10.5], [1, 1]],
        [['n14'], ['Spt'], ['n08'], [], [15.5, 10.5], [1, 1]],
        [['n15'], ['Spt'], ['n09'], ['n20', 'n09'], [15.5, 10.5], [1, 1]],
        [['n16'], ['Spt'], ['n10'], ['n21', 'n10'], [15.5, 10.5], [1, 1]],
        [['n17'], ['Spt'], ['n11'], [], [45.5, 20.5], [1, 1]],
        [['n18'], ['Spt'], ['n12'], ['n24', 'n12'], [45.5, 20.5], [1, 1]],
        [['n19'], ['Spt'], ['n13'], ['n25', 'n13'], [45.5, 20.5], [1, 1]],
        [['n20'], ['Dsc'], ['n14'], [], [2.5, 15.5], [1, 1]],
        [['n21'], ['Dsc'], ['n15'], [], [2.5, 15.5], [1, 1]],
        [['n22'], ['Dsc'], ['n16'], [], [2.5, 15.5], [1, 1]],
        [['n23'], ['Out'], ['n16'], [], [57.5, 15.5], [1, 1]],
        [['n24'], ['Dsc'], ['n17'], [], [2.5, 15.5], [1, 1]],
        [['n25'], ['Dsc'], ['n18'], [], [2.5, 15.5], [1, 1]],
        [['n26'], ['Dsc'], ['n19'], [], [2.5, 15.5], [1, 1]],
        [['n27'], ['Out'], ['n19'], [], [57.5, 15.5], [1, 1]],
    ]

    sg_covid_pcr = [
        [['n00'], ['Dis'], [], [], [15.5, 2.5], []],
        [['n01'], ['Dis'], [], [], [45.5, 2.5], [1, 1]],
        [['n02'], ['Dis'], [], [], [15.5, 27.5], [1, 1]],
        [['n03'], ['Dis'], [], [], [45.5, 27.5], [1, 1]],
        [['n04'], ['Mix'], ['n00', 'n01'], [], [15.5, 20.5], [1, 1]],
        [['n05'], ['Mix'], ['n02', 'n03'], [], [45.5, 10.5], [1, 1]],
        [['n06'], ['Spt'], ['n04'], [], [25.5, 10.5], [1, 1]],
        [['n07'], ['Spt'], ['n05'], ['n08', 'n09'], [25.5, 10.5], [1, 1]],
        [['n08'], ['Dsc'], ['n06'], [], [2.5, 15.5], [1, 1]],
        [['n09'], ['Mix'], ['n06', 'n07'], ['n06'], [15.5, 20.5], [1, 1]],
        [['n10'], ['Dsc'], ['n07'], [], [2.5, 15.5], [1, 1]],
        [['n11'], ['Spt'], ['n09'], ['n10', 'n09'], [25.5, 10.5], [1, 1]],
        [['n12'], ['Dsc'], ['n11'], [], [2.5, 15.5], [1, 1]],
        [['n13'], ['Thm'], ['n11'], [], [15.5, 10.5], [2, 1]],
        [['n14'], ['Thm'], ['n13'], [], [45.5, 20.5], [1, 1]],
        [['n15'], ['Thm'], ['n14'], ['n14'], [15.5, 10.5], [1, 1]],
        [['n16'], ['Thm'], ['n15'], ['n15'], [45.5, 20.5], [1, 1]],
        [['n17'], ['Thm'], ['n16'], ['n16'], [15.5, 10.5], [1, 1]],
        [['n18'], ['Thm'], ['n17'], ['n17'], [45.5, 20.5], [1, 1]],
        [['n19'], ['Thm'], ['n18'], ['n18'], [15.5, 10.5], [1, 1]],
        [['n20'], ['Thm'], ['n19'], ['n19'], [45.5, 20.5], [1, 1]],
        [['n21'], ['Thm'], ['n20'], ['n20'], [15.5, 10.5], [1, 1]],
        [['n22'], ['Thm'], ['n21'], ['n21'], [45.5, 20.5], [1, 1]],
        [['n23'], ['Thm'], ['n22'], ['n22'], [15.5, 10.5], [1, 1]],
        [['n24'], ['Thm'], ['n23'], ['n23'], [45.5, 20.5], [1, 1]],
        [['n25'], ['Thm'], ['n24'], ['n24'], [15.5, 10.5], [1, 1]],
        [['n26'], ['Thm'], ['n25'], ['n25'], [45.5, 20.5], [1, 1]],
        [['n27'], ['Thm'], ['n26'], ['n26'], [15.5, 10.5], [1, 1]],
        [['n28'], ['Thm'], ['n27'], ['n27'], [45.5, 20.5], [1, 1]],
        [['n29'], ['Thm'], ['n28'], ['n28'], [15.5, 10.5], [1, 1]],
        [['n30'], ['Thm'], ['n29'], ['n29'], [45.5, 20.5], [1, 1]],
        [['n31'], ['Thm'], ['n30'], ['n30'], [15.5, 10.5], [1, 1]],
        [['n32'], ['Thm'], ['n31'], ['n31'], [45.5, 20.5], [1, 1]],
        [['n33'], ['Out'], ['n32'], [], [57.5, 15.5], [1, 1]],
    ]

    sg_cp = [
        [['a01'], ['Dis'], [], [], [17.5, 2.5]],
        [['a02'], ['Dis'], [], [], [47.5, 2.5]],
        [['a03'], ['Dis'], [], [], [58.5, 24.5]],
        [['a04'], ['Dis'], [], [], [58.5, 9.5]],
        [['a05'], ['Mix'], ['a01', 'a02'], [], [34.5, 6.5]],
        [['a06'], ['Mix'], ['a03', 'a04'], [], [56.5, 18.5]],
        [['a07'], ['Mix'], ['a05', 'a06'], [], [52.5, 6.5]],
        [['b02'], ['Dis'], [], ['a07'], [17.5, 28.5]],
        [['b03'], ['Mix'], ['a07', 'b02'], [], [18.5, 6.5]],
        [['b04'], ['Mag'], ['b03'], [], [23.5, 6.5]],
        [['b05'], ['Spt'], ['b04'], [], [23.5, 6.5]],
        [['c02'], ['Dis'], [], ['b03'], [47.5, 2.5]],
        [['c03'], ['Mix'], ['b05', 'c02'], [], [34.5, 6.5]],
        [['c04'], ['Dis'], [], ['c03'], [47.5, 28.5]],
        [['c05'], ['Dlt'], ['c03', 'c04'], [], [49.5, 10.5]],
    ]

    sg_nuip8 = [
        [['n00'], ['Dis'], [], [], [15.5, 2.5], []],
        [['n01'], ['Dis'], [], [], [45.5, 2.5], [1, 1]],
        [['n02'], ['Dis'], [], [], [45.5, 27.5], [1, 1]],
        [['n03'], ['Dis'], [], ['n13'], [45.5, 27.5], [1, 1]],
        [['n04'], ['Dis'], [], ['n14'], [45.5, 27.5], [1, 1]],
        [['n05'], ['Dis'], [], ['n15'], [45.5, 27.5], [1, 1]],
        [['n06'], ['Dis'], [], ['n16'], [45.5, 27.5], [1, 1]],
        [['n07'], ['Dis'], [], ['n17'], [45.5, 27.5], [1, 1]],
        [['n08'], ['Dis'], [], ['n18'], [45.5, 27.5], [1, 1]],
        [['n09'], ['Dis'], [], ['n19'], [45.5, 27.5], [1, 1]],
        [['n10'], ['Dis'], [], [], [15.5, 27.5], [1, 1]],
        [['n11'], ['Dis'], [], ['n20'], [45.5, 27.5], [1, 1]],
        [['n12'], ['Mix'], ['n00', 'n01'], [], [15.5, 20.5], [1, 1]],
        [['n13'], ['Mix'], ['n32', 'n02'], [], [45.5, 10.5], [1, 1]],
        [['n14'], ['Mix'], ['n33', 'n03'], [], [45.5, 20.5], [2, 1]],
        [['n15'], ['Mix'], ['n34', 'n04'], ['n23'], [15.5, 20.5], [2, 1]],
        [['n16'], ['Mix'], ['n35', 'n05'], ['n24'], [45.5, 10.5], [2, 1]],
        [['n17'], ['Mix'], ['n36', 'n06'], ['n25'], [45.5, 20.5], [2, 1]],
        [['n18'], ['Mix'], ['n37', 'n07'], ['n26'], [15.5, 20.5], [2, 1]],
        [['n19'], ['Mix'], ['n38', 'n08'], ['n27'], [45.5, 10.5], [2, 1]],
        [['n20'], ['Mix'], ['n39', 'n09'], ['n28'], [45.5, 20.5], [2, 1]],
        [['n21'], ['Mix'], ['n20', 'n10'], ['n29'], [15.5, 20.5], [1, 1]],
        [['n22'], ['Mix'], ['n21', 'n11'], ['n30'], [45.5, 10.5], [1, 1]],
        [['n23'], ['Mag'], ['n12'], [], [15.5, 10.5], [1, 1]],
        [['n24'], ['Mag'], ['n13'], ['n32'], [15.5, 10.5], [1, 1]],
        [['n25'], ['Mag'], ['n14'], ['n33'], [15.5, 10.5], [1, 1]],
        [['n26'], ['Mag'], ['n15'], ['n34'], [15.5, 10.5], [1, 1]],
        [['n27'], ['Mag'], ['n16'], ['n35'], [15.5, 10.5], [1, 1]],
        [['n28'], ['Mag'], ['n17'], ['n36'], [15.5, 10.5], [1, 1]],
        [['n29'], ['Mag'], ['n18'], ['n37'], [15.5, 10.5], [1, 1]],
        [['n30'], ['Mag'], ['n19'], ['n38'], [15.5, 10.5], [1, 1]],
        [['n31'], ['Mag'], ['n22'], ['n39'], [15.5, 10.5], [1, 1]],
        [['n32'], ['Spt'], ['n23'], [], [25.5, 10.5], [1, 1]],
        [['n33'], ['Spt'], ['n24'], ['n13', 'n41'], [25.5, 10.5], [1, 1]],
        [['n34'], ['Spt'], ['n25'], ['n14', 'n42'], [25.5, 10.5], [1, 1]],
        [['n35'], ['Spt'], ['n26'], ['n15', 'n43'], [25.5, 10.5], [1, 1]],
        [['n36'], ['Spt'], ['n27'], ['n16', 'n44'], [25.5, 10.5], [1, 1]],
        [['n37'], ['Spt'], ['n28'], ['n17', 'n45'], [25.5, 10.5], [1, 1]],
        [['n38'], ['Spt'], ['n29'], ['n18', 'n46'], [25.5, 10.5], [1, 1]],
        [['n39'], ['Spt'], ['n30'], ['n19', 'n47'], [25.5, 10.5], [1, 1]],
        [['n40'], ['Spt'], ['n31'], ['n20', 'n48'], [25.5, 10.5], [1, 1]],
        [['n41'], ['Dsc'], ['n32'], [], [2.5, 15.5], [1, 1]],
        [['n42'], ['Dsc'], ['n33'], [], [2.5, 15.5], [1, 1]],
        [['n43'], ['Dsc'], ['n34'], [], [2.5, 15.5], [1, 1]],
        [['n44'], ['Dsc'], ['n35'], [], [2.5, 15.5], [1, 1]],
        [['n45'], ['Dsc'], ['n36'], [], [2.5, 15.5], [1, 1]],
        [['n46'], ['Dsc'], ['n37'], [], [2.5, 15.5], [1, 1]],
        [['n47'], ['Dsc'], ['n38'], [], [2.5, 15.5], [1, 1]],
        [['n48'], ['Dsc'], ['n39'], [], [2.5, 15.5], [1, 1]],
        [['n49'], ['Out'], ['n40'], [], [57.5, 15.5], [2, 1]],
        [['n50'], ['Dsc'], ['n40'], [], [2.5, 15.5], [1, 1]],
    ]

    # [FIXME] Check MasterMix last argument
    sg_master_mix = [
        [['n00'], ['Dis'], [], [], [15.5, 2.5], [1, 1]],
        [['n01'], ['Dis'], [], [], [45.5, 2.5], [1, 1]],
        [['n02'], ['Dis'], [], [], [15.5, 28.5], [1, 1]],
        [['n03'], ['Dis'], [], [], [45.5, 28.5], [1, 1]],
        [['n04'], ['Dis'], [], ['n20'], [15.5, 2.5], [1, 1]],
        [['n05'], ['Dis'], [], ['n21'], [45.5, 2.5], [1, 1]],
        [['n06'], ['Dis'], [], ['n21'], [15.5, 28.5], [1, 1]],
        [['n07'], ['Dis'], [], ['n22'], [45.5, 28.5], [1, 1]],
        [['n08'], ['Dis'], [], ['n23'], [15.5, 2.5], [1, 1]],
        [['n09'], ['Dis'], [], ['n24'], [45.5, 2.5], [1, 1]],
        [['n10'], ['Dis'], [], ['n24'], [15.5, 28.5], [1, 1]],
        [['n11'], ['Dis'], [], ['n25'], [45.5, 28.5], [1, 1]],
        [['n12'], ['Dis'], [], ['n26'], [15.5, 2.5], [1, 1]],
        [['n13'], ['Dis'], [], ['n27'], [45.5, 2.5], [1, 1]],
        [['n14'], ['Dis'], [], ['n27'], [15.5, 28.5], [1, 1]],
        [['n15'], ['Dis'], [], ['n28'], [45.5, 28.5], [1, 1]],
        [['n16'], ['Dis'], [], ['n29'], [15.5, 2.5], [1, 1]],
        [['n17'], ['Dis'], [], ['n30'], [45.5, 2.5], [1, 1]],
        [['n18'], ['Dis'], [], ['n30'], [15.5, 28.5], [1, 1]],
        [['n19'], ['Dis'], [], ['n31'], [45.5, 28.5], [1, 1]],
        [['n20'], ['Mix'], ['n00', 'n21'], [], [15.5, 20.5], [1, 1]],
        [['m21'], ['Mix'], ['n01', 'n02'], [], [15.5, 10.5], [1, 1]],
        [['n21'], ['Spt'], ['m21'], [], [15.5, 10.5], [1, 1]],
        [['n22'], ['Mix'], ['n21', 'n03'], [], [45.5, 20.5], [1, 1]],
        [['n23'], ['Mix'], ['n04', 'n24'], [], [45.5, 20.5], [1, 1]],
        [['m24'], ['Mix'], ['n05', 'n06'], ['n20', 'n22'], [15.5, 10.5], [1, 1]],
        [['n24'], ['Spt'], ['m24'], [], [15.5, 10.5], [1, 1]],
        [['n25'], ['Mix'], ['n24', 'n07'], ['n35'], [45.5, 10.5], [1, 1]],
        [['n26'], ['Mix'], ['n08', 'n27'], ['n36'], [45.5, 10.5], [1, 1]],
        [['m27'], ['Mix'], ['n09', 'n10'], ['n23', 'n25'], [15.5, 10.5], [1, 1]],
        [['n27'], ['Spt'], ['m27'], [], [15.5, 10.5], [1, 1]],
        [['n28'], ['Mix'], ['n27', 'n11'], ['n38'], [15.5, 20.5], [1, 1]],
        [['n29'], ['Mix'], ['n12', 'n30'], ['n38'], [15.5, 20.5], [1, 1]],
        [['m30'], ['Mix'], ['n13', 'n14'], ['n26', 'n28'], [15.5, 10.5], [1, 1]],
        [['n30'], ['Spt'], ['m30'], [], [15.5, 10.5], [1, 1]],
        [['n31'], ['Mix'], ['n30', 'n15'], ['n39'], [45.5, 20.5], [1, 1]],
        [['n32'], ['Mix'], ['n16', 'n33'], ['n40'], [45.5, 20.5], [1, 1]],
        [['m33'], ['Mix'], ['n17', 'n18'], ['n29', 'n31'], [15.5, 10.5], [1, 1]],
        [['n33'], ['Spt'], ['m33'], [], [15.5, 10.5], [1, 1]],
        [['n34'], ['Mix'], ['n33', 'n19'], ['n41'], [45.5, 10.5], [1, 1]],
        [['n35'], ['Out'], ['n20'], [], [57.5, 15.5], [1, 1]],
        [['n36'], ['Out'], ['n22'], [], [57.5, 15.5], [1, 1]],
        [['n37'], ['Out'], ['n23'], [], [57.5, 15.5], [1, 1]],
        [['n38'], ['Out'], ['n25'], [], [57.5, 15.5], [1, 1]],
        [['n39'], ['Out'], ['n26'], [], [57.5, 15.5], [1, 1]],
        [['n40'], ['Out'], ['n28'], [], [57.5, 15.5], [1, 1]],
        [['n41'], ['Out'], ['n29'], [], [57.5, 15.5], [1, 1]],
        [['n42'], ['Out'], ['n31'], [], [57.5, 15.5], [1, 1]],
        [['n43'], ['Out'], ['n32'], [], [57.5, 15.5], [1, 1]],
        [['n44'], ['Out'], ['n34'], [], [57.5, 15.5], [1, 1]],
    ]

    # [FIXME] Check SD last argument
    sg_sd = [
        [['n00'], ['Dis'], [], [], [15.5, 2.5], [1, 1]],
        [['n01'], ['Dis'], [], [], [15.5, 28.5], [1, 1]],
        [['n02'], ['Dis'], [], [], [45.5, 28.5], [1, 1]],
        [['n03'], ['Dis'], [], ['n16'], [15.5, 2.5], [1, 1]],
        [['n04'], ['Dis'], [], [], [45.5, 2.5], [1, 1]],
        [['n05'], ['Dis'], [], ['n17'], [15.5, 28.5], [1, 1]],
        [['n06'], ['Dis'], [], ['n18'], [45.5, 28.5], [1, 1]],
        [['n07'], ['Dis'], [], ['n19'], [45.5, 2.5], [1, 1]],
        [['n08'], ['Dis'], [], ['n20'], [15.5, 28.5], [1, 1]],
        [['n09'], ['Dis'], [], ['n21'], [45.5, 28.5], [1, 1]],
        [['n10'], ['Dis'], [], ['n22'], [45.5, 2.5], [1, 1]],
        [['n11'], ['Dis'], [], ['n23'], [15.5, 28.5], [1, 1]],
        [['n12'], ['Dis'], [], ['n24'], [45.5, 28.5], [1, 1]],
        [['n13'], ['Dis'], [], ['n25'], [45.5, 2.5], [1, 1]],
        [['n14'], ['Dis'], [], ['n26'], [15.5, 28.5], [1, 1]],
        [['n15'], ['Dis'], [], ['n27'], [45.5, 28.5], [1, 1]],
        [['m16'], ['Mag'], ['n00'], [], [15.5, 20.5], [1, 1]],
        [['n16'], ['Spt'], ['m16'], [], [15.5, 20.5], [1, 1]],
        [['n17'], ['Mix'], ['n16', 'n01'], [], [45.5, 10.5], [1, 1]],
        [['n18'], ['Mix'], ['n16', 'n02'], [], [45.5, 20.5], [1, 1]],
        [['m19'], ['Mix'], ['n03', 'n04'], [], [15.5, 10.5], [1, 1]],
        [['n19'], ['Spt'], ['m19'], [], [15.5, 10.5], [1, 1]],
        [['n20'], ['Mix'], ['n33', 'n05'], ['n31'], [45.5, 10.5], [1, 1]],
        [['n21'], ['Mix'], ['n33', 'n06'], ['n32'], [45.5, 20.5], [1, 1]],
        [['m22'], ['Mix'], ['n19', 'n07'], ['n33'], [15.5, 10.5], [1, 1]],
        [['n22'], ['Spt'], ['m22'], [], [15.5, 10.5], [1, 1]],
        [['n23'], ['Mix'], ['n36', 'n08'], ['n34'], [45.5, 10.5], [1, 1]],
        [['n24'], ['Mix'], ['n36', 'n09'], ['n35'], [45.5, 20.5], [1, 1]],
        [['m25'], ['Mix'], ['n22', 'n10'], ['n36'], [15.5, 10.5], [1, 1]],
        [['n25'], ['Spt'], ['m25'], [], [15.5, 10.5], [1, 1]],
        [['n26'], ['Mix'], ['n39', 'n11'], ['n37'], [45.5, 10.5], [1, 1]],
        [['n27'], ['Mix'], ['n39', 'n12'], ['n38'], [45.5, 20.5], [1, 1]],
        [['m28'], ['Mix'], ['n25', 'n13'], ['n39'], [15.5, 10.5], [1, 1]],
        [['n28'], ['Spt'], ['m28'], [], [15.5, 10.5], [1, 1]],
        [['n29'], ['Mix'], ['n42', 'n14'], ['n40'], [45.5, 10.5], [1, 1]],
        [['n30'], ['Mix'], ['n42', 'n15'], ['n41'], [45.5, 20.5], [1, 1]],
        [['n31'], ['Out'], ['n17'], [], [58.5, 15.5], [1, 1]],
        [['n32'], ['Out'], ['n18'], [], [58.5, 15.5], [1, 1]],
        [['m33'], ['Mag'], ['n19'], ['n17', 'n18'], [15.5, 20.5], [1, 1]],
        [['n33'], ['Spt'], ['m33'], [], [15.5, 20.5], [1, 1]],
        [['n34'], ['Out'], ['n20'], [], [58.5, 15.5], [1, 1]],
        [['n35'], ['Out'], ['n21'], [], [58.5, 15.5], [1, 1]],
        [['m36'], ['Mag'], ['n22'], ['n20', 'n21'], [15.5, 20.5], [1, 1]],
        [['n36'], ['Spt'], ['m36'], [], [15.5, 20.5], [1, 1]],
        [['n37'], ['Out'], ['n23'], [], [58.5, 15.5], [1, 1]],
        [['n38'], ['Out'], ['n24'], [], [58.5, 15.5], [1, 1]],
        [['m39'], ['Mag'], ['n25'], ['n23', 'n24'], [15.5, 20.5], [1, 1]],
        [['n39'], ['Spt'], ['m39'], [], [15.5, 20.5], [1, 1]],
        [['n40'], ['Out'], ['n26'], [], [58.5, 15.5], [1, 1]],
        [['n41'], ['Out'], ['n27'], [], [58.5, 15.5], [1, 1]],
        [['m42'], ['Mag'], ['n28'], ['n26', 'n27'], [15.5, 20.5], [1, 1]],
        [['n42'], ['Spt'], ['m42'], [], [15.5, 20.5], [1, 1]],
        [['n43'], ['Out'], ['n29'], [], [58.5, 15.5], [1, 1]],
        [['n44'], ['Out'], ['n30'], [], [58.5, 15.5], [1, 1]],
    ]