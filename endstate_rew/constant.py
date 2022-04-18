from openmm import unit
from openmmtools.constants import kB

### define units
distance_unit = unit.angstrom
time_unit = unit.femtoseconds
speed_unit = distance_unit / time_unit

# constants
stepsize = 1 * time_unit
collision_rate = 1 / unit.picosecond
temperature = 300 * unit.kelvin

kBT = kB * temperature

zinc_systems = [
    ("ZINC61095", "CCOc1ccc2nc(/N=C\c3ccccc3O)sc2c1"),
    ("ZINC77329", "Cn1cc(Cl)c(/C=N/O)n1"),
    ("ZINC79729", "S=c1cc(-c2ccc(Cl)cc2)ss1"),
    ("ZINC86442", "CN1C(=O)C/C(=N\O)N(C)C1=O"),
    ("ZINC87557", "NNC(=O)[C@H]1C(c2ccccc2)[C@@H]1C(=O)NN"),
    ("ZINC95858", "CCO/C(O)=N/S(=O)(=O)c1ccccc1Cl"),
    ("ZINC107550", "C/C(=N\O)c1oc(C)nc1C"),
    ("ZINC107778", "O/N=C/C1=C(Cl)c2cc(Cl)ccc2OC1"),
    ("ZINC123162", "CC(=O)/C(=N/Nc1ccc(Cl)cc1)C(=O)c1ccccc1"),
    ("ZINC133435", "c1ccc(-c2nc3ccccc3nc2-c2ccccn2)nc1"),
    ("ZINC138607", "O=C(CC1=NO[C@H](c2ccccc2O)N1)N1CCCC1"),
    ("ZINC140610", "Cc1cc(C)c2c(=O)[nH]sc2n1"),
    ("ZINC164361", "CCON1C(=O)c2ccccc2C1=O"),
    ("ZINC167648", "Cc1ccc(COn2c(-c3ccccc3)nc3ccccc3c2=O)cc1"),
    ("ZINC169358", "CC1=Cn2c(=O)c3ccccc3c(=O)n2C1"),
    ("ZINC1036618", "COc1ccc(/C=C2/N=C(c3ccccc3)NNC2=O)cc1"),
    ("ZINC1755198", "CC(C)C(=O)NNC(=O)C(C)C"),
    ("ZINC1867000", "c1ccc(-c2ccccc2-c2ccccc2)cc1"),
    ("ZINC3127671", "O=C(CSCC(=O)Nc1ccccc1)NNC(=O)c1ccccc1"),
    ("ZINC4344392", "CCOC(=O)NNC(=O)NCCCc1ccc2ccc3cccc4ccc1c2c34"),
    ("ZINC4363792", "Clc1cc(Cl)cc(/N=c2\ssnc2-c2ccccc2Cl)c1"),
    ("ZINC6568023", "O=C(NNC(=O)c1ccccc1)c1ccccc1"),
    ("ZINC33381936", "O=S(=O)(O/N=C1/CCc2ccccc21)c1ccc(Cl)cc1"),
]
