"""Deterministic proton stopping-power ratio (SPR) from the DoseRAD Geant4 phantom.

The DoseRAD2026 Geant4 simulator (``geant4-dose-sim`` ``DICOMphantom.cc``,
``InitialisationOfMaterials``) defines, for every HU material, the *exact*
elemental mass fractions **and explicitly sets the mean excitation energy**
(``GetIonisation()->SetMeanExcitationEnergy``). The proton electronic stopping
power the MC actually used is therefore fully determined by data we can read
off: composition + I + density. We reproduce the water-relative SPR with the
same inputs Geant4's Bethe-Bloch electronic stopping used:

    SPR(E) = (rho/rho_w)
             * (sum_i w_i Z_i/A_i)_mat / (sum_i w_i Z_i/A_i)_w
             * [ ln(2 m_e c^2 beta^2 gamma^2 / I_mat) - beta^2 ]
             / [ ln(2 m_e c^2 beta^2 gamma^2 / I_w  ) - beta^2 ]

The only terms dropped vs full Geant4 transport are shell / density-effect
corrections (largely cancelling in a water ratio for tissue/bone at
therapeutic energies) and nuclear interactions (not part of WEPL). The WEPL is
therefore essentially exact with no free parameters -- this removes the
RSP/range identifiability confound and lets a differentiable LUT calibrate
pure base-data error.

beta^2 is evaluated once at the nominal beam energy (constant along the ray);
the log-ratio varies sub-percent over the therapeutic range.

Material order == ``DICOMphantom.cc`` ``fOriginalMaterials`` push order ==
definition order == the index produced by
``pydose_rt.physics.materials.geant4_material_id_from_hu``. Auto-generated from
the upstream source; do not hand-edit the table.
"""

from __future__ import annotations

import numpy as np
import torch

from pydose_rt.physics.materials import GEANT4_HU_BOUNDS, GEANT4_NUM_MATERIALS

# Element (Z, A g/mol) exactly as declared in DICOMphantom.cc.
_ELEMENTS: dict[str, tuple[float, float]] = {

    'C': (6.0, 12.011),
    'H': (1.0, 1.008),
    'N': (7.0, 14.007),
    'O': (8.0, 16.0),
    'Na': (11.0, 22.98977),
    'S': (16.0, 32.065),
    'Cl': (17.0, 35.453),
    'K': (19.0, 39.0983),
    'P': (15.0, 30.973976),
    'Mg': (12.0, 24.305),
    'Ca': (20.0, 40.078),
}

# (I_eV, nominal_density_g_cm3, ((element, mass_fraction), ...)) per material,
# in Geant4 fOriginalMaterials / material-id order (86 materials).
G4_MATERIALS: tuple[tuple[float, float, tuple[tuple[str, float], ...]], ...] = (
    (87.10503, 0.00121, (('N', 0.76494), ('O', 0.23506))),  # HU_1024to_950
    (75.17784, 0.26, (('H', 0.103), ('C', 0.105), ('N', 0.031), ('O', 0.749), ('Na', 0.002), ('P', 0.002), ('S', 0.003), ('Cl', 0.003), ('K', 0.002))),  # HU_950to_90
    (63.86476, 0.9528, (('H', 0.11527), ('C', 0.64204), ('N', 0.0044), ('O', 0.23716), ('Na', 0.00114))),  # HU_90to_64
    (65.91813, 0.9787, (('H', 0.11258), ('C', 0.53768), ('N', 0.0107), ('O', 0.33664), ('S', 0.0012), ('Cl', 0.0012))),  # HU_64to_38
    (64.69939, 0.9926, (('H', 0.11401), ('C', 0.60012), ('N', 0.00936), ('O', 0.27425), ('S', 0.00113), ('Cl', 0.00113))),  # HU_38to_24
    (66.50713, 1.001, (('H', 0.11173), ('C', 0.5077), ('N', 0.01425), ('O', 0.36356), ('S', 0.00138), ('Cl', 0.00138))),  # HU_24to_10
    (68.32664, 1.01, (('H', 0.1095), ('C', 0.41751), ('N', 0.01902), ('O', 0.4507), ('S', 0.00163), ('Cl', 0.00163))),  # HU_10to4
    (70.15735, 1.024, (('H', 0.10732), ('C', 0.32949), ('N', 0.02368), ('O', 0.53575), ('S', 0.00188), ('Cl', 0.00188))),  # HU4to18
    (74.60653, 1.076, (('H', 0.10412), ('C', 0.12394), ('N', 0.0266), ('O', 0.73521), ('Na', 0.00172), ('P', 0.00191), ('S', 0.00203), ('Cl', 0.00219), ('K', 0.00228))),  # HU18to70
    (74.63808, 1.11, (('H', 0.09925), ('C', 0.17298), ('N', 0.03887), ('O', 0.67473), ('Na', 0.00317), ('P', 0.00434), ('S', 0.00384), ('Cl', 0.00284))),  # HU70to120
    (72.70762, 1.108, (('H', 0.09604), ('C', 0.46025), ('N', 0.02456), ('O', 0.35443), ('P', 0.01966), ('S', 0.00178), ('Cl', 0.00122), ('Ca', 0.04206))),  # HU120to140
    (73.54896, 1.121, (('H', 0.09428), ('C', 0.45162), ('N', 0.02506), ('O', 0.35675), ('P', 0.02203), ('S', 0.00182), ('Cl', 0.00119), ('Ca', 0.04726))),  # HU140to160
    (74.38228, 1.135, (('H', 0.09257), ('C', 0.44319), ('N', 0.02555), ('O', 0.35901), ('P', 0.02434), ('S', 0.00185), ('Cl', 0.00115), ('Ca', 0.05233))),  # HU160to180
    (75.20764, 1.148, (('H', 0.0909), ('C', 0.43496), ('N', 0.02602), ('O', 0.36122), ('P', 0.0266), ('S', 0.00189), ('Cl', 0.00112), ('Ca', 0.05729))),  # HU180to200
    (76.02511, 1.162, (('H', 0.08926), ('C', 0.42693), ('N', 0.02648), ('O', 0.36338), ('P', 0.02881), ('S', 0.00192), ('Cl', 0.00109), ('Ca', 0.06213))),  # HU200to220
    (76.83475, 1.175, (('H', 0.08767), ('C', 0.41908), ('N', 0.02694), ('O', 0.36549), ('P', 0.03096), ('S', 0.00195), ('Cl', 0.00106), ('Ca', 0.06686))),  # HU220to240
    (77.63663, 1.188, (('H', 0.08611), ('C', 0.41141), ('N', 0.02738), ('O', 0.36755), ('P', 0.03307), ('S', 0.00198), ('Cl', 0.00103), ('Ca', 0.07148))),  # HU240to260
    (78.3732, 1.202, (('H', 0.08467), ('C', 0.40432), ('N', 0.02784), ('O', 0.36993), ('P', 0.03516), ('S', 0.00201), ('Ca', 0.07607))),  # HU260to280
    (79.1615, 1.215, (('H', 0.08318), ('C', 0.39697), ('N', 0.02826), ('O', 0.37189), ('P', 0.03717), ('S', 0.00204), ('Ca', 0.08049))),  # HU280to300
    (79.94222, 1.229, (('H', 0.08172), ('C', 0.38978), ('N', 0.02867), ('O', 0.37381), ('P', 0.03914), ('S', 0.00207), ('Ca', 0.08481))),  # HU300to320
    (80.71542, 1.242, (('H', 0.08029), ('C', 0.38275), ('N', 0.02908), ('O', 0.37568), ('P', 0.04106), ('S', 0.0021), ('Ca', 0.08904))),  # HU320to340
    (81.48118, 1.255, (('H', 0.07889), ('C', 0.37588), ('N', 0.02947), ('O', 0.37752), ('P', 0.04295), ('S', 0.00213), ('Ca', 0.09317))),  # HU340to360
    (82.23957, 1.269, (('H', 0.07753), ('C', 0.36915), ('N', 0.02986), ('O', 0.37931), ('P', 0.04479), ('S', 0.00215), ('Ca', 0.09722))),  # HU360to380
    (82.99067, 1.282, (('H', 0.07619), ('C', 0.36256), ('N', 0.03024), ('O', 0.38107), ('P', 0.04659), ('S', 0.00218), ('Ca', 0.10118))),  # HU380to400
    (83.73455, 1.296, (('H', 0.07488), ('C', 0.35612), ('N', 0.03061), ('O', 0.38279), ('P', 0.04836), ('S', 0.00221), ('Ca', 0.10505))),  # HU400to420
    (84.47128, 1.309, (('H', 0.07359), ('C', 0.3498), ('N', 0.03097), ('O', 0.38447), ('P', 0.05009), ('S', 0.00223), ('Ca', 0.10885))),  # HU420to440
    (85.20094, 1.323, (('H', 0.07234), ('C', 0.34362), ('N', 0.03132), ('O', 0.38612), ('P', 0.05178), ('S', 0.00226), ('Ca', 0.11257))),  # HU440to460
    (85.98254, 1.336, (('H', 0.07103), ('C', 0.33721), ('N', 0.03164), ('O', 0.38734), ('Mg', 0.00103), ('P', 0.05339), ('S', 0.00228), ('Ca', 0.11609))),  # HU460to480
    (86.69996, 1.349, (('H', 0.06982), ('C', 0.33127), ('N', 0.03198), ('O', 0.38891), ('Mg', 0.00106), ('P', 0.05501), ('S', 0.0023), ('Ca', 0.11965))),  # HU480to500
    (87.41049, 1.363, (('H', 0.06864), ('C', 0.32545), ('N', 0.03231), ('O', 0.39045), ('Mg', 0.00109), ('P', 0.0566), ('S', 0.00232), ('Ca', 0.12314))),  # HU500to520
    (88.11419, 1.376, (('H', 0.06748), ('C', 0.31974), ('N', 0.03264), ('O', 0.39195), ('Mg', 0.00112), ('P', 0.05816), ('S', 0.00235), ('Ca', 0.12656))),  # HU520to540
    (88.81115, 1.39, (('H', 0.06634), ('C', 0.31414), ('N', 0.03296), ('O', 0.39343), ('Mg', 0.00115), ('P', 0.05968), ('S', 0.00237), ('Ca', 0.12991))),  # HU540to560
    (89.50144, 1.403, (('H', 0.06523), ('C', 0.30866), ('N', 0.03327), ('O', 0.39488), ('Mg', 0.00118), ('P', 0.06118), ('S', 0.00239), ('Ca', 0.1332))),  # HU560to580
    (90.18513, 1.417, (('H', 0.06414), ('C', 0.30328), ('N', 0.03358), ('O', 0.3963), ('Mg', 0.00121), ('P', 0.06265), ('S', 0.00241), ('Ca', 0.13643))),  # HU580to600
    (90.86231, 1.43, (('H', 0.06306), ('C', 0.298), ('N', 0.03388), ('O', 0.3977), ('Mg', 0.00124), ('P', 0.06409), ('S', 0.00243), ('Ca', 0.13959))),  # HU600to620
    (91.53304, 1.443, (('H', 0.06201), ('C', 0.29282), ('N', 0.03417), ('O', 0.39906), ('Mg', 0.00127), ('P', 0.06551), ('S', 0.00245), ('Ca', 0.1427))),  # HU620to640
    (92.19739, 1.457, (('H', 0.06098), ('C', 0.28774), ('N', 0.03447), ('O', 0.40041), ('Mg', 0.0013), ('P', 0.0669), ('S', 0.00247), ('Ca', 0.14574))),  # HU640to660
    (92.85545, 1.47, (('H', 0.05996), ('C', 0.28275), ('N', 0.03475), ('O', 0.40173), ('Mg', 0.00132), ('P', 0.06826), ('S', 0.00249), ('Ca', 0.14873))),  # HU660to680
    (93.50728, 1.484, (('H', 0.05897), ('C', 0.27785), ('N', 0.03503), ('O', 0.40302), ('Mg', 0.00135), ('P', 0.0696), ('S', 0.00251), ('Ca', 0.15167))),  # HU680to700
    (94.15295, 1.497, (('H', 0.05799), ('C', 0.27305), ('N', 0.0353), ('O', 0.40429), ('Mg', 0.00137), ('P', 0.07091), ('S', 0.00253), ('Ca', 0.15455))),  # HU700to720
    (94.79254, 1.511, (('H', 0.05703), ('C', 0.26833), ('N', 0.03557), ('O', 0.40554), ('Mg', 0.0014), ('P', 0.0722), ('S', 0.00255), ('Ca', 0.15738))),  # HU720to740
    (95.42612, 1.524, (('H', 0.05609), ('C', 0.26369), ('N', 0.03584), ('O', 0.40676), ('Mg', 0.00142), ('P', 0.07346), ('S', 0.00257), ('Ca', 0.16016))),  # HU740to760
    (96.05376, 1.537, (('H', 0.05517), ('C', 0.25913), ('N', 0.0361), ('O', 0.40796), ('Mg', 0.00145), ('P', 0.07471), ('S', 0.00259), ('Ca', 0.16289))),  # HU760to780
    (96.67553, 1.551, (('H', 0.05426), ('C', 0.25466), ('N', 0.03636), ('O', 0.40915), ('Mg', 0.00147), ('P', 0.07593), ('S', 0.00261), ('Ca', 0.16557))),  # HU780to800
    (97.2915, 1.564, (('H', 0.05336), ('C', 0.25026), ('N', 0.03661), ('O', 0.41031), ('Mg', 0.0015), ('P', 0.07713), ('S', 0.00262), ('Ca', 0.16821))),  # HU800to820
    (97.90173, 1.578, (('H', 0.05248), ('C', 0.24594), ('N', 0.03685), ('O', 0.41145), ('Mg', 0.00152), ('P', 0.07831), ('S', 0.00264), ('Ca', 0.1708))),  # HU820to840
    (98.5063, 1.591, (('H', 0.05162), ('C', 0.24169), ('N', 0.0371), ('O', 0.41257), ('Mg', 0.00154), ('P', 0.07947), ('S', 0.00266), ('Ca', 0.17335))),  # HU840to860
    (99.10527, 1.604, (('H', 0.05077), ('C', 0.23752), ('N', 0.03734), ('O', 0.41368), ('Mg', 0.00156), ('P', 0.08061), ('S', 0.00267), ('Ca', 0.17585))),  # HU860to880
    (99.6987, 1.618, (('H', 0.04994), ('C', 0.23341), ('N', 0.03757), ('O', 0.41476), ('Mg', 0.00158), ('P', 0.08173), ('S', 0.00269), ('Ca', 0.17831))),  # HU880to900
    (100.2867, 1.631, (('H', 0.04912), ('C', 0.22938), ('N', 0.0378), ('O', 0.41583), ('Mg', 0.00161), ('P', 0.08283), ('S', 0.00271), ('Ca', 0.18073))),  # HU900to920
    (100.8692, 1.645, (('H', 0.04831), ('C', 0.22541), ('N', 0.03803), ('O', 0.41688), ('Mg', 0.00163), ('P', 0.08392), ('S', 0.00272), ('Ca', 0.18311))),  # HU920to940
    (101.4465, 1.658, (('H', 0.04752), ('C', 0.2215), ('N', 0.03825), ('O', 0.41791), ('Mg', 0.00165), ('P', 0.08498), ('S', 0.00274), ('Ca', 0.18545))),  # HU940to960
    (102.0184, 1.672, (('H', 0.04674), ('C', 0.21766), ('N', 0.03847), ('O', 0.41892), ('Mg', 0.00167), ('P', 0.08603), ('S', 0.00275), ('Ca', 0.18775))),  # HU960to980
    (102.5852, 1.685, (('H', 0.04597), ('C', 0.21388), ('N', 0.03869), ('O', 0.41992), ('Mg', 0.00169), ('P', 0.08707), ('S', 0.00277), ('Ca', 0.19002))),  # HU980to1000
    (103.1468, 1.698, (('H', 0.04521), ('C', 0.21016), ('N', 0.0389), ('O', 0.4209), ('Mg', 0.00171), ('P', 0.08808), ('S', 0.00278), ('Ca', 0.19225))),  # HU1000to1020
    (103.7033, 1.712, (('H', 0.04447), ('C', 0.2065), ('N', 0.03911), ('O', 0.42187), ('Mg', 0.00173), ('P', 0.08908), ('S', 0.0028), ('Ca', 0.19444))),  # HU1020to1040
    (104.2548, 1.725, (('H', 0.04374), ('C', 0.2029), ('N', 0.03931), ('O', 0.42282), ('Mg', 0.00175), ('P', 0.09006), ('S', 0.00281), ('Ca', 0.1966))),  # HU1040to1060
    (104.8013, 1.739, (('H', 0.04302), ('C', 0.19935), ('N', 0.03952), ('O', 0.42376), ('Mg', 0.00177), ('P', 0.09103), ('S', 0.00283), ('Ca', 0.19873))),  # HU1060to1080
    (105.343, 1.752, (('H', 0.04231), ('C', 0.19586), ('N', 0.03972), ('O', 0.42468), ('Mg', 0.00179), ('P', 0.09199), ('S', 0.00284), ('Ca', 0.20082))),  # HU1080to1100
    (105.8797, 1.766, (('H', 0.04161), ('C', 0.19243), ('N', 0.03991), ('O', 0.42559), ('Mg', 0.0018), ('P', 0.09292), ('S', 0.00285), ('Ca', 0.20288))),  # HU1100to1120
    (106.4117, 1.779, (('H', 0.04092), ('C', 0.18904), ('N', 0.04011), ('O', 0.42648), ('Mg', 0.00182), ('P', 0.09385), ('S', 0.00287), ('Ca', 0.20491))),  # HU1120to1140
    (106.939, 1.792, (('H', 0.04024), ('C', 0.18571), ('N', 0.0403), ('O', 0.42736), ('Mg', 0.00184), ('P', 0.09476), ('S', 0.00288), ('Ca', 0.20691))),  # HU1140to1160
    (107.4616, 1.806, (('H', 0.03958), ('C', 0.18242), ('N', 0.04048), ('O', 0.42823), ('Mg', 0.00186), ('P', 0.09566), ('S', 0.00289), ('Ca', 0.20888))),  # HU1160to1180
    (107.9795, 1.819, (('H', 0.03892), ('C', 0.17919), ('N', 0.04067), ('O', 0.42909), ('Mg', 0.00187), ('P', 0.09654), ('S', 0.00291), ('Ca', 0.21082))),  # HU1180to1200
    (108.4929, 1.833, (('H', 0.03827), ('C', 0.176), ('N', 0.04085), ('O', 0.42993), ('Mg', 0.00189), ('P', 0.09741), ('S', 0.00292), ('Ca', 0.21273))),  # HU1200to1220
    (109.0018, 1.846, (('H', 0.03763), ('C', 0.17286), ('N', 0.04103), ('O', 0.43076), ('Mg', 0.00191), ('P', 0.09827), ('S', 0.00293), ('Ca', 0.21461))),  # HU1220to1240
    (109.5062, 1.86, (('H', 0.037), ('C', 0.16977), ('N', 0.04121), ('O', 0.43157), ('Mg', 0.00192), ('P', 0.09911), ('S', 0.00294), ('Ca', 0.21647))),  # HU1240to1260
    (110.0063, 1.873, (('H', 0.03638), ('C', 0.16672), ('N', 0.04138), ('O', 0.43238), ('Mg', 0.00194), ('P', 0.09995), ('S', 0.00296), ('Ca', 0.21829))),  # HU1260to1280
    (110.5019, 1.886, (('H', 0.03577), ('C', 0.16371), ('N', 0.04155), ('O', 0.43317), ('Mg', 0.00196), ('P', 0.10077), ('S', 0.00297), ('Ca', 0.22009))),  # HU1280to1300
    (110.9933, 1.9, (('H', 0.03517), ('C', 0.16075), ('N', 0.04172), ('O', 0.43396), ('Mg', 0.00197), ('P', 0.10157), ('S', 0.00298), ('Ca', 0.22187))),  # HU1300to1320
    (111.4805, 1.913, (('H', 0.03458), ('C', 0.15783), ('N', 0.04189), ('O', 0.43473), ('Mg', 0.00199), ('P', 0.10237), ('S', 0.00299), ('Ca', 0.22362))),  # HU1320to1340
    (112.0059, 1.927, (('H', 0.03396), ('C', 0.1548), ('N', 0.04201), ('O', 0.43505), ('Na', 0.001), ('Mg', 0.002), ('P', 0.10306), ('S', 0.003), ('Ca', 0.22512))),  # HU1340to1360
    (112.4846, 1.94, (('H', 0.03338), ('C', 0.15196), ('N', 0.04217), ('O', 0.4358), ('Na', 0.001), ('Mg', 0.00202), ('P', 0.10383), ('S', 0.00301), ('Ca', 0.22682))),  # HU1360to1380
    (112.9592, 1.953, (('H', 0.03281), ('C', 0.14916), ('N', 0.04233), ('O', 0.43654), ('Na', 0.00101), ('Mg', 0.00203), ('P', 0.10459), ('S', 0.00302), ('Ca', 0.2285))),  # HU1380to1400
    (113.4297, 1.967, (('H', 0.03225), ('C', 0.1464), ('N', 0.04249), ('O', 0.43727), ('Na', 0.00101), ('Mg', 0.00205), ('P', 0.10535), ('S', 0.00303), ('Ca', 0.23015))),  # HU1400to1420
    (113.8962, 1.98, (('H', 0.0317), ('C', 0.14368), ('N', 0.04265), ('O', 0.43798), ('Na', 0.00102), ('Mg', 0.00206), ('P', 0.10609), ('S', 0.00305), ('Ca', 0.23178))),  # HU1420to1440
    (114.3588, 1.994, (('H', 0.03115), ('C', 0.141), ('N', 0.0428), ('O', 0.43869), ('Na', 0.00102), ('Mg', 0.00207), ('P', 0.10682), ('S', 0.00306), ('Ca', 0.23339))),  # HU1440to1460
    (114.8175, 2.007, (('H', 0.03062), ('C', 0.13835), ('N', 0.04295), ('O', 0.43939), ('Na', 0.00102), ('Mg', 0.00209), ('P', 0.10754), ('S', 0.00307), ('Ca', 0.23497))),  # HU1460to1480
    (115.2722, 2.021, (('H', 0.03009), ('C', 0.13574), ('N', 0.0431), ('O', 0.44008), ('Na', 0.00103), ('Mg', 0.0021), ('P', 0.10826), ('S', 0.00308), ('Ca', 0.23654))),  # HU1480to1500
    (115.7232, 2.034, (('H', 0.02956), ('C', 0.13316), ('N', 0.04325), ('O', 0.44076), ('Na', 0.00103), ('Mg', 0.00212), ('P', 0.10896), ('S', 0.00309), ('Ca', 0.23808))),  # HU1500to1520
    (116.1704, 2.047, (('H', 0.02905), ('C', 0.13062), ('N', 0.04339), ('O', 0.44143), ('Na', 0.00103), ('Mg', 0.00213), ('P', 0.10965), ('S', 0.0031), ('Ca', 0.23961))),  # HU1520to1540
    (116.6139, 2.061, (('H', 0.02854), ('C', 0.12811), ('N', 0.04353), ('O', 0.44209), ('Na', 0.00104), ('Mg', 0.00214), ('P', 0.11034), ('S', 0.00311), ('Ca', 0.24111))),  # HU1540to1560
    (117.0536, 2.074, (('H', 0.02803), ('C', 0.12563), ('N', 0.04368), ('O', 0.44274), ('Na', 0.00104), ('Mg', 0.00216), ('P', 0.11101), ('S', 0.00312), ('Ca', 0.24259))),  # HU1560to1580
    (117.4897, 2.088, (('H', 0.02754), ('C', 0.12319), ('N', 0.04382), ('O', 0.44338), ('Na', 0.00104), ('Mg', 0.00217), ('P', 0.11168), ('S', 0.00313), ('Ca', 0.24406))),  # HU1580to1600
    (117.9222, 2.101, (('H', 0.02704), ('C', 0.12078), ('N', 0.04395), ('O', 0.44402), ('Na', 0.00105), ('Mg', 0.00218), ('P', 0.11234), ('S', 0.00314), ('Ca', 0.2455))),  # HU1600to1620
    (117.9222, 3.708, (('H', 0.02704), ('C', 0.12078), ('N', 0.04395), ('O', 0.44402), ('Na', 0.00105), ('Mg', 0.00218), ('P', 0.11234), ('S', 0.00314), ('Ca', 0.2455))),  # HU4000to4000
)


assert len(G4_MATERIALS) == GEANT4_NUM_MATERIALS, (
    f"SPR table has {len(G4_MATERIALS)} materials but "
    f"GEANT4_NUM_MATERIALS={GEANT4_NUM_MATERIALS}; the SPR table and the "
    "HU->material-id mapping have diverged."
)

# --- Physical constants ---
_ME_C2_EV: float = 0.51099895069e6      # electron rest energy [eV]
_MP_C2_MEV: float = 938.27208816        # proton rest energy [MeV]

# Liquid-water reference (Geant4 G4_WATER convention: I = 78 eV).
_WATER_I_EV: float = 78.0
_WATER_COMP: tuple[tuple[str, float], ...] = (("H", 2.0 * 1.008), ("O", 16.00))


def _zoa(comp: tuple[tuple[str, float], ...]) -> float:
    """Sum_i w_i Z_i / A_i  (mol electrons per gram) for normalised fractions."""
    total = sum(frac for _, frac in comp)
    return sum(
        (frac / total) * _ELEMENTS[sym][0] / _ELEMENTS[sym][1] for sym, frac in comp
    )


MATERIAL_I_EV: np.ndarray = np.asarray([m[0] for m in G4_MATERIALS], dtype=np.float64)
MATERIAL_DENSITY: np.ndarray = np.asarray([m[1] for m in G4_MATERIALS], dtype=np.float64)
MATERIAL_ZOA: np.ndarray = np.asarray(
    [_zoa(m[2]) for m in G4_MATERIALS], dtype=np.float64
)
WATER_ZOA: float = _zoa(_WATER_COMP)


def proton_beta2_gamma2(energy_mev: float) -> tuple[float, float]:
    """(beta^2, gamma^2) for a proton of kinetic energy ``energy_mev``."""
    gamma = 1.0 + float(energy_mev) / _MP_C2_MEV
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    return beta2, gamma * gamma


def _bethe_L(I_eV: torch.Tensor, beta2: float, gamma2: float) -> torch.Tensor:
    """Bethe stopping logarithm  ln(2 m_e c^2 beta^2 gamma^2 / I) - beta^2."""
    arg = (2.0 * _ME_C2_EV * beta2 * gamma2) / I_eV
    return torch.log(arg) - beta2


def spr_factor(
    material_id: torch.Tensor,
    energy_mev: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-voxel composition factor s.t. SPR = density * spr_factor.

    ``spr_factor = (zoa_mat / zoa_w) * L(I_mat) / L(I_w)`` -- the
    density-independent part of the water-relative Bethe SPR, evaluated at the
    (constant) nominal beam energy. Deterministic; carries no gradient.
    """
    if device is None:
        device = material_id.device
    beta2, gamma2 = proton_beta2_gamma2(energy_mev)
    mid = material_id.to(device=device, dtype=torch.long).clamp_(0, GEANT4_NUM_MATERIALS - 1)
    i_mat = torch.as_tensor(MATERIAL_I_EV, device=device, dtype=dtype)[mid]
    zoa_mat = torch.as_tensor(MATERIAL_ZOA, device=device, dtype=dtype)[mid]
    l_mat = _bethe_L(i_mat, beta2, gamma2)
    l_w = _bethe_L(
        torch.tensor(_WATER_I_EV, device=device, dtype=dtype), beta2, gamma2
    )
    return (zoa_mat / WATER_ZOA) * (l_mat / l_w)


def hu_to_material_id(hu: torch.Tensor) -> torch.Tensor:
    """HU -> Geant4 material index (same mapping as geant4_material_id_from_hu)."""
    bounds = torch.as_tensor(GEANT4_HU_BOUNDS, device=hu.device, dtype=hu.dtype)
    clipped = hu.clamp(float(GEANT4_HU_BOUNDS[0]), float(GEANT4_HU_BOUNDS[-1]))
    ids = torch.bucketize(clipped.contiguous(), bounds[1:].contiguous(), right=True)
    return ids.clamp_(0, GEANT4_NUM_MATERIALS - 1).long()


def density_from_hu(
    hu: torch.Tensor, hu_to_density_entries: list[dict[str, float]]
) -> torch.Tensor:
    """HU -> density via piecewise-linear interpolation, entirely on hu's device."""
    xs_np = np.asarray([float(e["hu"]) for e in hu_to_density_entries], dtype=np.float64)
    ys_np = np.asarray(
        [float(e["density_g_cm3"]) for e in hu_to_density_entries], dtype=np.float64
    )
    order = np.argsort(xs_np)
    xs_np, ys_np = xs_np[order], ys_np[order]
    xs = torch.as_tensor(xs_np, device=hu.device, dtype=hu.dtype)
    ys = torch.as_tensor(ys_np, device=hu.device, dtype=hu.dtype)
    x = hu.clamp(float(xs_np[0]), float(xs_np[-1]))
    idx = (
        torch.searchsorted(xs.contiguous(), x.contiguous(), right=True)
        .sub_(1)
        .clamp_(0, xs.shape[0] - 2)
    )
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    t = (x - x0) / (x1 - x0).clamp_min(1e-9)
    return y0 + t * (y1 - y0)


def spr_and_mass_density(
    hu: torch.Tensor,
    energy_mev: float,
    hu_to_density_entries: list[dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the engine's two volumes from a HU volume.

    Returns ``(spr_volume, mass_density_volume)``:
      * ``spr_volume``  = physical_density * spr_factor -> feed as the engine's
        ``density_image`` (drives WEPL / range).
      * ``mass_density_volume`` = physical density -> feed as
        ``mass_density_image`` (drives MeV->Gy, must stay physical).
    """
    mass_density = density_from_hu(hu, hu_to_density_entries)
    material_id = hu_to_material_id(hu)
    factor = spr_factor(material_id, energy_mev, device=hu.device, dtype=hu.dtype)
    return mass_density * factor, mass_density


def patient_dose_mask(
    mass_density: torch.Tensor,
    density_threshold_g_cm3: float = 0.03,
) -> torch.Tensor:
    """Voxels where dose is scored: everything except air open to the outside world.

    The DoseRAD reference is exactly zero outside the patient and nonzero inside
    internal air (trachea, bowel gas, sinuses) -- measured up to 83% of a beamlet's
    peak there, since dose to air under charged-particle equilibrium is comparable to
    the surrounding tissue. Density alone cannot tell the two apart: both sit at
    ~0.0012 g/cm3. So the split has to be topological, not a threshold.

    External air is the largest sub-threshold connected component that touches the
    volume border. Everything else -- tissue, and any enclosed air pocket -- is scored.
    Note that an internal cavity may itself touch the border (a trachea cut by the
    first slice), which is why this takes the largest such component rather than all
    of them.

    Falls back to the plain threshold if that component holds less than half of the
    sub-threshold voxels, i.e. the geometry is not the expected "patient surrounded by
    air" (a FOV that the patient fills, a phantom run).
    """
    from scipy import ndimage

    sub = (mass_density <= float(density_threshold_g_cm3)).detach().cpu().numpy()
    if not sub.any():
        return torch.ones_like(mass_density, dtype=torch.bool)

    labels, _ = ndimage.label(sub)
    border = np.zeros_like(sub, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    border[:, :, 0] = border[:, :, -1] = True
    border_labels = labels[border & sub]
    if border_labels.size == 0:
        return torch.ones_like(mass_density, dtype=torch.bool)

    ids, counts = np.unique(border_labels, return_counts=True)
    external = labels == ids[int(np.argmax(counts))]
    if external.sum() < 0.5 * sub.sum():
        return mass_density > float(density_threshold_g_cm3)

    # Connectivity alone still zeroes cavities that VENT to the outside -- a trachea
    # reaching the pharynx, bowel gas reaching the rectum. Measured on 1THB002: 12,224
    # such voxels, up to 2.1% of a beamlet's integral and 488 scored-core voxels, and it
    # is why that case did not move when only the enclosed cavities were kept.
    # Union rather than replacement: this can only ever add to what connectivity keeps,
    # so detached limbs and anything else already scored cannot be lost.
    body = _body_contour(mass_density, density_threshold_g_cm3)
    return torch.from_numpy(~external | body).to(device=mass_density.device)


def _body_contour(
    mass_density: torch.Tensor,
    density_threshold_g_cm3: float,
    min_component_voxels: int = 5000,
) -> np.ndarray:
    """Patient exterior, as a per-axial-slice hole fill of the above-threshold region.

    Per-slice, not 3D: a 3D fill cannot close a lumen that is open at a z face (a trachea
    cut by the first slice), while in-plane that same lumen is enclosed by tissue. Runs in
    ~1 s on a 33M-voxel volume -- SimpleITK's 3D fill filter is the slow one.

    Components below ``min_component_voxels`` are dropped as noise; everything above is
    kept, so a limb detached from the trunk by the FOV survives as its own component.
    """
    from scipy import ndimage

    solid = (mass_density > float(density_threshold_g_cm3)).detach().cpu().numpy()
    filled = np.stack([ndimage.binary_fill_holes(s) for s in solid])
    labels, n = ndimage.label(filled)
    if n == 0:
        return filled
    sizes = ndimage.sum(filled, labels, range(1, n + 1))
    keep = [i + 1 for i, size in enumerate(sizes) if size >= min_component_voxels]
    return np.isin(labels, keep)
