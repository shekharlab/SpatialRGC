import numpy as np
#Format of each[[XL,XU], [YL,YU]]# Units are in microns
RN_REGIONS = {
    "140g_rn3_rg0":[
        np.array([[2750,3000],[10000,10250]]),
        np.array([[2890,2915],[10030,10055]]),
        np.array([[2890,2915],[10030,10055]]),
        np.array([[3250,3500],[10100,10350]]),
        np.array([[2750,2800],[10060,10110]]),#Example bipolar vs amacrine,
        np.array([[2750,3000],[10125,10250]]),# Example of undersizing alphas,
        np.array([[3000,3300],[10400,10700]]),# artery,
        np.array([[3250,3500],[9400,9650]]),# artery,
        np.array([[3600,3900],[10700,11000]]),# bad artery,
        np.array([[3300,3900],[10400,11000]]),# bad artery BIG,
        np.array([[2875,3000],[10125,10250]]),#Figure 1
        np.array([[2750,3000],[10125,10375]]),# Example of undersizing alphas/F-rgc correct size,
        ],
    "140g_rn3_rg1":[
        np.array([[9550,9850],[8650,8950]]), # used z7
        np.array([[9975,10225],[7675,7925]]), # used z7; Also in figure 1; also for F-RGC
        np.array([[9975,10225],[7675,7925]]),
        np.array([[10000,10150],[7700,7850]]),
        np.array([[10000,10300],[7700,8000]]),
        np.array([[11200,11450], [8100,8350]]), # Temporal sample for alpha,
        np.array([[10075,10150], [7675,7750]]), #Qual,
        np.array([[10400,10500], [8900,8975]]),
        np.array([[10400,10550], [8900,9050]]), #Artery
        np.array([[9400,9700],[8600,8900]]),#Non-neuronal,
        np.array([[9400,9700],[8000,8300]]),#Non-neuronal,
        np.array([[9400,9700],[9000,9300]]),#Non-neuronal,
        np.array([[11500,12000],[7800,8300]]),#11 Non-neuronal,
        np.array([[9975,10037],[7735,7755]]),
        ],
    "140g_rn3_rg2":[
        np.array([[350,650], [3250,3550]]) # Nasal example
    ],
    "140g_rn3_rg3":[
        np.array([[10000,10250],[2500,2750]]),
        ], # VERY BAD EXAMPLE OF THE STAINING
    "140g_rn4_rg0":[
        np.array([[4450,4700],[12125,12375]]),
        np.array([[3000,3250],[11250,11500]]),#Not segmented non-neuronal
        np.array([[4150,4300],[10000,10150]]),#GOOD NON-NEURONAL FIG
        np.array([[4225,4262],[10102,10122]]),#debug white space
        np.array([[4243,4250],[10108,10115]])#debug white space

        ],
    "140g_rn4_rg2":[
        np.array([[1450,1600],[3450,3600]]),#GOOD NON-NEURONAL STAIN EXAMPLE

    ],
    "140g_rn4_rg3":[
        np.array([[8000, 8250], [750, 1000]]),
        np.array([[8900, 9150], [1200, 1450]]),
        np.array([[7270,7560],[675,975]]),#vein non-neuronal #GOOD NON-NEURONAL FIG
        np.array([[8000, 8300], [250, 550]]),#outer non-neuronal (not segmented)
        np.array([[8000, 8250], [400, 650]]),#outer non-neuronal (not segmented),
        np.array([[7325, 7475], [775, 925]]),#outer non-neuronal (not segmented)

    ],
    "140g_rn6_rg1":[
        np.array([[2500,2750],[9500,9750]])
    ],
    "140g_rn6_rg4":[
        np.array([[515,765],[2825,3075]]),
    ],
    "140g_rn8_rg0":[],
    "140g_rn8_rg1":[],
    "140g_rn8_rg2":[],
    "140g_rn8_rg3":[],
    "140g_rn8_rg4":[
        np.array([[11700, 12000],[200,500]])
    ],
    "140g_rn14_rg1":[
        np.array([[8340,8560],[3336,3556]]),#bad
        np.array([[10325, 10545],[841,  1061]]),#1 example
        np.array([[10151, 10381],#eh 1 example
       [  856,  1015]]),
       np.array([[10853, 11149],
       [ 1168,  1425]]),
       np.array([[10500,10545],[895,940]])
    ],
    "140g_rn16_rg0":[
        np.array([[11117.42862434, 11765.40987287],
        [ 3429.45675889, 4077.44054348]])
    ],
    "140g_rn16_rg1":[
        np.array([[3410.82756221, 3553.38337816],
       [3152.54609471, 3295.10409836]]),
       np.array([[2513.15790907, 3290.73508696],
                [2196.11148838, 2800.90301903]]) #C10,C24
    ],
    "140g_rn16_rg2":[
        np.array([[10915.98136415, 11063.71892814],
       [16991.38735385, 17130.48565236]])
    ],
    "140g_rn16_rg3":[
        np.array([[10922.89306305, 11048.16760561],
       [ 8608.33896596,  8733.61383108]]),
       np.array([[10915.98136415, 11063.71892814],
       [ 8351.74148362,  8490.83978213]]) #GOOD
    ],
    "140g_rn17_rg0":[
      np.array([[10945.60488045, 11312.79629682],[2517.85663943, 2885.05111913]])  #C33 example
    ],
    "140g_rn17_rg1":[
       np.array([[8373.62912449, 9021.61150756],
                [2572.49146209, 3220.47891732]]) #C31, C33 sparse example
    ],
    "140g_rn17_rg4":[
        np.array([[9473.81576702, 9701.90396842],
       [8154.38464719, 8385.06312065]]),
       np.array([[9448.32863845, 9663.45728296],
       [6800.55255762, 7026.91120949]])
    ],
    "140g_rn26_rg2":[
        np.array([[11929, 12174],[ 8180,  8382]]),
        np.array([[11659 , 11851],[ 8901,  9042]]), #Nmb,Opn4,
        np.array([[11000,11300],[8000,8300]]),#Example region
        np.array([[11500,11800],[8500,8800]]),
        np.array([[12115, 12275],[ 8200,  8360]]),#GOOD
        np.array([[11540, 11700],[7590, 7750]]), #GOOD


    ],
    "140g_rn26_rg4":[
      
    ],
    "202303151505_Fresh-frozen-C57BL6-2-3mo_VMSC01801_rg4":[
        np.array([[5500,5800],[6000,6300]]),#for visual
        np.array([[6000,6300],[6000,6300]]),#for visual
        np.array([[6500,6800],[6100,6400]]),#for visual,
        np.array([[5800,6000],[6000,6200]]),#for visual

        np.array([[5500,5650],[6000,6150]]),#1 split
        np.array([[5650,5800],[6000,6150]]),
        np.array([[5500,5650],[6150,6300]]),
        np.array([[5650,5800],[6150,6300]]),

        np.array([[6000,6150],[6000,6150]]),#for visual
        np.array([[6150,6300],[6000,6150]]),
        np.array([[6000,6150],[6150,6300]]),
        np.array([[6150,6300],[6150,6300]]),

        np.array([[6500,6650],[6100,6250]]),
        np.array([[6650,6800],[6100,6250]]),
        np.array([[6500,6650],[6250,6400]]),
        np.array([[6650,6800],[6250,6400]]),

    ],
    "140g_rn100_rg4":[
        np.array([[5500,5800],[6000,6300]]),#for visual
        np.array([[6000,6300],[6000,6300]]),#for visual
        np.array([[6500,6800],[6100,6400]]),#for visual,
        np.array([[5800,6000],[6000,6200]]),#for visual

        np.array([[5500,5650],[6000,6150]]),#1 split
        np.array([[5650,5800],[6000,6150]]),
        np.array([[5500,5650],[6150,6300]]),
        np.array([[5650,5800],[6150,6300]]),

        np.array([[6000,6150],[6000,6150]]),#for visual
        np.array([[6150,6300],[6000,6150]]),
        np.array([[6000,6150],[6150,6300]]),
        np.array([[6150,6300],[6150,6300]]),

        np.array([[6500,6650],[6100,6250]]),
        np.array([[6650,6800],[6100,6250]]),
        np.array([[6500,6650],[6250,6400]]),
        np.array([[6650,6800],[6250,6400]]),

    ],
    "202303131740_PanWGA4-P4-inj-slide2_VMSC01801_rg4":[
        np.array([[1500,1800],[6000,6300]]),#for visual
        np.array([[2500,2800],[6500,6800]]),
        np.array([[1500,1800],[5500,5800]]),
        np.array([[3000,3300],[7000,7300]]),
        np.array([[3000,3300],[7000,7300]]),

        np.array([[3025,3100],[7150,7225]]),#for train
        np.array([[2975,3100],[7125,7250]]),
        np.array([[1500,1625],[6000,6125]]),
        np.array([[1625,1750],[6000,6125]]),
        np.array([[1750,1875],[6000,6125]]),
        np.array([[1500,1625],[6125,6250]]),
        np.array([[1625,1750],[6125,6250]]),
        np.array([[1750,1875],[6125,6250]]),

        np.array([[2500,2625],[6500,6625]]),#for train more
        np.array([[2500,2625],[6625,6750]]),
        np.array([[2500,2625],[6750,6875]]),

        np.array([[2625,2750],[6500,6625]]),
        np.array([[2625,2750],[6625,6750]]),
        np.array([[2625,2750],[6750,6875]]),

        np.array([[1500,2500],[5500,6500]]),#19, to chunk
        np.array([[2500,3500],[6000,7000]]),#20, to chunk,
        np.array([[1500,1700], [5500, 5700]])#Example for excitatory/inhibitory


    ],
    "140g_rn101_rg4":[
        np.array([[1500,1800],[6000,6300]]),#for visual
        np.array([[2500,2800],[6500,6800]]),
        np.array([[1500,1800],[5500,5800]]),
        np.array([[3000,3300],[7000,7300]]),
        np.array([[3000,3300],[7000,7300]]),

        np.array([[3025,3100],[7150,7225]]),#for train
        np.array([[2975,3100],[7125,7250]]),
        np.array([[1500,1625],[6000,6125]]),
        np.array([[1625,1750],[6000,6125]]),
        np.array([[1750,1875],[6000,6125]]),
        np.array([[1500,1625],[6125,6250]]),
        np.array([[1625,1750],[6125,6250]]),
        np.array([[1750,1875],[6125,6250]]),

        np.array([[2500,2625],[6500,6625]]),#for train more
        np.array([[2500,2625],[6625,6750]]),
        np.array([[2500,2625],[6750,6875]]),

        np.array([[2625,2750],[6500,6625]]),
        np.array([[2625,2750],[6625,6750]]),
        np.array([[2625,2750],[6750,6875]]),

    ],
    "140g_rn102_rg0":[
        np.array([[2000,3250],[10400,10800]]),#big training sample
    ],
    "140g_rn102_rg4":[
        np.array([[12000,13250],[4500,5000]]),#big training sample,
        np.array([[12600,12800],[4700,4900]]),#for visual,
    ],
    "140g_rn103_rg3":[
        np.array([[10500,12000],[7000,7800]]),#big training sample,
    ],
    "140g_rn105_rg1":[
        np.array([[1500,3000],[8500,9500]]),#big training sample,
    ],
    "140g_rn105_rg2":[
        np.array([[6000,7250],[7250,7750]]),#big training sample,
    ],
    "140g_rn105_rg3":[
        np.array([[11000,12000],[5750,6500]]),#big training sample,
    ],
    "140g_rn105_rg4":[
        np.array([[1000,2250],[4000,4750]]),#big training sample,
        np.array([[1750,2000],[4250,4500]]),#big training sample,


    ],
    "140g_rn105_rg6":[
        np.array([[10000,11500],[2000,3000]]),#big training sample,
    ],
    "140g_rn106_rg0":[
        np.array([[2000,3000],[2500,3250]]),#big training sample,
    ],
    "140g_rn106_rg1":[
        np.array([[9500,10500],[1500,2500]]),#big training sample,
    ],
    "140g_rn106_rg2":[
        np.array([[12500,13500],[2000,3000]]),#big training sample,
    ],
    "140g_rn106_rg3":[
        np.array([[12500,13500],[2000,3000]]),#big training sample,
    ],
    "140g_rn106_rg7":[
        np.array([[12750,13750],[6000,7000]]),#big training sample,
    ],
    "140g_rn201_rg1":[
        np.array([[9000,9250],[7625,7875]]),#visualization example
        np.array([[8800,9600],[7500,8100]]),#big training sample,
    ],
    "140g_rn202_rg1":[
        np.array([[6000,6250],[2250,2500]]),#visualization example
        np.array([[5900,6500],[2000,2600]]),#big training sample,
    ],
    "140g_rn202_rg2":[
        np.array([[2600,2850],[2500,2750]]),#visualization, example
        np.array([[2600,3000],[2500,2900]]),#training_example
    ],
    "140g_rn202_rg4":[
        np.array([[6000,6250],[6625,6875]]),#visualization example,
        np.array([[6000,6400],[6400,6800]])#bigtraining sample
    ],
     "140g_rn203_rg1":[
        np.array([[1750,2000],[8500,8750]]),#visualization example,
        np.array([[1750,2000],[8600,8850]])
    ],

}
# [(x1,y1)...] where order of coordinate pairs is counterclockwise
BIPOLAR_ZONES = {
    "140g_rn3_rg0":[
        [(1790,10595),(2325,10075),(2300,10600),(1790,10600)],
    ],
    "140g_rn3_rg1":[
        [(9419,9784),(9615,9482),(9941,9238),(10023,9251),(9464,9948),(9419,9784)],
        [(9617,7816),(9793,8007),(9755,8073),(9568,7885),(9617,7816)]
    ],
    "140g_rn3_rg2":[
        [(805,3645),(1068,3288),(1492,3057),(1557,2851),(2089,2872),(2124,3027),(2255,3003),(2235,2792),(3203,3283),(2900,4290),(1119,4035),(805,3645)],
        [(1165,1627),(2308,2040),(2789,2209),(2447,2575),(2122,2569),(2213,2403),(1963,2275),(1693,2584),(1216,2596),(1165,1627)]
    ],
    "140g_rn3_rg3":[
        [(9136,722),(10651,1053),(11545,1964),(10803,1677),(10263,1818),(10140,1758),(10108,1627),(9906,1640),(9802,1702),(9706,1855),(9869,2108),(10366,1877),(10955,1864),(11351,2495),(10901,3607),(10196,3916),(8682,2957),(9136,722)]
        ],
    "140g_rn4_rg0":[
        [(4561,10425),(4722,10423),(4577,10749),(4456,10648),(4561,10425)],
    ],
    "140g_rn4_rg1":[],
    "140g_rn4_rg2":[
        [(1273,4285),(1240,4661),(1624,4643),(1738,4365),(2162,4282),(2184,4614),(2797,4049),(2370,3291),(1273,4285)],
    ],
    "140g_rn4_rg3":[],
    "140g_rn5_rg0":[],
    "140g_rn5_rg1":[],
    "140g_rn5_rg2":[[(2248,3188),(2364,3024),(2544,3126),(2402,3574),(2211,3520),(2248,3188)],
                    [(2002,3836),(2183,3952),(2044,4152),(1889,4055),(2002,3836)]
    ],
    "140g_rn5_rg3":[
        [(7693,1722),(8561,963),(9429,820),(10051,1006),(10453,3123),(9514,3416),(8502,3381),(7849,2855),(8510,2111),(8669,2096),(8888,2373),(9055,2329),(9213,1996),(9122,1884),(8815,1878),(8666,2046),(8066,2077),(7693,1722)]
    ], 
    "140g_rn6_rg1":[
        [(873,8808),(1535,8146),(1593,8599),(898,8843),(873,8808)],
        [(2264,8197),(2733,8008),(2765,8212),(2251,8387),(2264,8197)]
    ],
    "140g_rn6_rg2":[
        [(6248,6701),(7431,6271),(7829,7247),(8798,7454),(9006,7685),(9246,7720),(9291,7891),(9099,7926),(8862,7996),(8959,9232),(8391,9697),(7387,10062),(6608,9683),(6194,9389),(5864,8434),(6056,6743),(6248,6701)],
    ],
    "140g_rn6_rg4":[
        [(724,2204),(1175,1919),(1285,2072),(813,2249),(700,2271),(724,2204)]
    ],
    "140g_rn7_rg0":[
        [(7788,9124),(7884,9122),(7865,9235),(7777,9244),(7713,9203),(7708,9115),(7788,9124)],
        [(8985,9004),(9029,8936),(9144,8958),(9166,9043),(9056,9066),(8992,9046),(8963,9012),(8985,9004)]
    ],
    "140g_rn7_rg1":[
        [(11597,8189),(12472,8001),(12842,7884),(13106,8247),(13109,8369),(13051,8507),(13928,8645),(13551,9499),(12990,9995),(12952,10443),(12343,9656),(11416,8950),(11483,8226),(11597,8189)],
    ],
    "140g_rn7_rg2":[],
    "140g_rn7_rg3":[
        [(1127,1802),(1659,1305),(3562,1430),(4160,2238),(3806,3229),(3269,4059),(2545,3903),(1350,3241),(1127,1802)]
    ],
    "140g_rn7_rg4":[
        [(9812,477),(12058,978),(12727,1966),(12904,2348),(12498,2998),(11583,3891),(10985,4024),(9863,3762),(9143,2876),(9812,477)]
    ],
    "140g_rn8_rg0":[
        [(6842,6269),(8661,6155),(8870,7046),(9503,7434),(9499,8027),(8781,8737),(8947,9832),(7255,9820),(6730,8176),(6842,6269)],
        [(9806,10766),(9762,10872),(9836,11098),(9948,10841),(9914,10692),(9806,10766)],
        [(9531,9436),(9464,9605),(9733,9695),(9657,9495),(9531,9436)],
        [(10229,9208),(10400,9471),(10400,9718),(10712,9648),(10593,9495),(10320,9032),(10229,9208)],
        [(9259,9853),(9282,10525),(9598,10493),(9675,10573),(9721,10509),(9557,9849),(9259,9853)]
    ],
    "140g_rn8_rg1":[
        [(1181,4599),(854,6825),(1328,8545),(1923,8762),(2086,9025),(3823,9223),(4133,9231),(4007,8163),(3278,7771),(3233,7245),(4000,6601),(3948,5804),(2475,4859),(1567,4454),(1181,4599)],
        [(4479,7411),(4529,7589),(4630,7925),(4917,7961),(4863,7763),(4810,7581),(4768,7486),(4628,7297),(4465,7328),(4479,7411)],
        [(3605,7431),(3607,7814),(4403,8028),(4417,7771),(4147,7561),(4164,7099),(3934,7091),(3697,7312),(3612,7427),(3605,7431)]
    ],
    "140g_rn8_rg2":[[(12319,5100),(13806,4854),(14618,5987),(14981,5879),(14885,6646),(14409,6763),(13898,6544),(13650,6908),(12552,6714),(12319,5100)]],
    "140g_rn8_rg3":[[(8933,2437),(8997,2512),(9067,2604),(9122,2686),(9176,2758),(9275,2712),(9240,2394),(9030,2309),(8933,2289),(8900,2358),(8933,2437)]],
    "140g_rn8_rg4":[[(11905,3676),(11940,3839),(12517,3721),(12494,3433),(12008,3442),(11905,3676)]],
    "140g_rn9_rg0":[
        [(11610,1628),(13728,1133),(14539,2542),(14513,3359),(13810,3830),(12136,3981),(11460,3466),(11610,1628)]
    ],
    "140g_rn9_rg1":[
        [(6589,1335),(6028,2842),(6405,3406),(8612,3230),(9269,2452),(8450,871),(6589,1335)],
    ],
    "140g_rn9_rg2":[
        [(1266,2231),(1654,2424),(2210,2104),(2069,1874),(1892,1795),(1893,1536),(2052,1189),(1809,1004),(996,1729),(1266,2231)],
        [(2427,1787),(2355,2784),(3736,2419),(3181,955),(2501,770),(2316,1442),(2676,1584),(2570,1752),(2427,1787)]
    ],
    "140g_rn9_rg3":[
        []
    ],
    "140g_rn9_rg4":[
        []
    ],
    "140g_rn10_rg0":[
        [(2078,2769),(2217,2769),(2223,3360),(3581,3888),(3566,4441),(2301,4953),(1254,4073),(1289,3498),(1887,2833),(2078,2769)]
    ],
    "140g_rn10_rg1":[
        [(8106,2518),(8700,2849),(7941,2743),(7873,2490),(8106,2518)]
    ],
    "140g_rn10_rg2":[
        []
    ],
    "140g_rn10_rg3":[
        []
    ],
    "140g_rn10_rg4":[
        []
    ],
    "140g_rn11_rg0":[
        [(11775,7062),(12058,6500),(12431,7184),(12216,7668),(11929,7688),(11775,7062)]
    ],
    "140g_rn11_rg1":[
        []
    ],
    "140g_rn11_rg2":[# Excluded
        []
    ],
    "140g_rn11_rg3":[# Excluded
        []
    ],
    "140g_rn11_rg4":[
        [(6318,3003),(6689,3331),(6675,3409),(6273,3087),(6272,2990),(6318,3003)]
    ],
    "140g_rn11_rg5":[
        []
    ],
    "140g_rn12_rg0":[
        [(2312,8380),(3060,7874),(4115,8185),(4024,9996),(3132,10128),(2450,9418),(2312,8380)]
    ],
    "140g_rn12_rg1":[
        [(8460,8029),(8930,8050),(9000,8858),(8672,8849),(8460,8029)]
    ],
    "140g_rn12_rg2":[
        []
    ],
    "140g_rn12_rg3":[
        []
    ],
    "140g_rn12_rg4":[
        [(6717,1102),(8635,713),(8663,2353),(7396,2303),(6717,1102)]
    ],
    "140g_rn12_rg5":[
        [(453,2575),(1462,498),(2754,453),(3185,2442),(3086,2951),(1854,2939),(498,3118),(453,2575)]
    ],
    "140g_rn14_rg0":[
        [(1985,49),(2811,1068),(4212,762),(4950,408),(4673,-18),(1985,-0.05),(1985,49)]
    ],
    "140g_rn14_rg1":[
        [(8123,2612),(8792,2428),(8782,2582),(8125,2775),(8123,2612)],
        [(9333,2953),(9511,2953),(9571,2642),(9434,2636),(9331,2835),(9333,2953)],
        [(10021,2920),(10098,3367),(10334,3376),(10159,2866),(10036,2890),(10021,2920)],
        [(10044,1959),(10558,1696),(10889,1506),(11053,1467),(10988,1633),(10710,1823),(10094,2122),(9912,2086),(10044,1959)],
        [(9660,857),(9789,869),(9737,1198),(9616,1183),(9602,844),(9660,857)]
    ],
    "140g_rn14_rg2":[
        [(10672,8375),(10678,8246),(10778,8246),(10734,8418),(10672,8375)],
        [(11422,8449),(11488,8703),(11603,8781),(11648,8658),(11482,8309),(11420,8380),(11422,8449)],
        [(11366,7414),(11644,7243),(11848,7102),(11995,7011),(12113,6960),(12179,6951),(12155,7054),(11556,7420),(11418,7474),(11349,7466),(11366,7414)]
    ],
    "140g_rn14_rg3":[
        [(5775,10775),(5801,10950),(5883,11022),(5896,10760),(5749,10763)],
        [(5851,9728),(6391,9368),(6385,9303),(6211,9358),(6029,9488),(5908,9603),(5708,9711),(5723,9767),(5809,9752)]
    ],
    "140g_rn14_rg4":[
        [(962,5515),(1709,5222),(1713,5428),(929,5717),(863,5581)],
        [(1263,5776),(1417,5609),(1468,5686),(1324,5874),(1211,5828),(1263,5775)],
        [(1826,6582),(1996,6079),(2178,6104),(2002,6669),(1826,6582)],
        [(2166,6027),(2235,5285),(2666,5299),(2437,6139),(2166,6027)],
        [(1994,4859),(2134,4852),(2097,5114),(1923,5002),(1994,4859)],
        [(1984,3799),(2249,4810),(2714,4713),(2638,3479),(1883,3583),(1984,3799)],
        [(2866,5626),(3145,6233),(3364,6006),(3010,5462),(2866,5626)],
        [(2795,4636),(3765,3918),(3993,4333),(3303,5222),(2787,5159),(2795,4636)]
    ],
    "140g_rn15_rg0":[
        [(2785,10130),(3939,10140),(3892,11022),(2700,10798),(2785,10130)],
        [(2166,9532),(2764,8065),(3143,8079),(2579,9770),(2166,9532)]
    ],
    "140g_rn15_rg1":[
        [(6063,9694),(8160,9127),(8300,8490),(7297,6558),(7001,6937),(6813,8358),(6646,8542),(6228,8354),(6063,9694)],
        [(5993,7080),(6154,6944),(6185,7254),(5964,7345),(5993,7080)],

    ],
    "140g_rn15_rg2":[
        []
    ],
    "140g_rn15_rg3":[
        []
    ],
    "140g_rn15_rg4":[
        []
    ],
    "140g_rn15_rg5":[
        [(1163,5657),(2884,5458),(3601,4270),(3346,3104),(2345,1575),(1276,1684),(475,3046),(696,5121),(1163,5657)]
    ],
    "140g_rn16_rg0":[
        [(9749,2607),(10377,3264),(11824,3026),(13047,1585),(12842,915),(11755,245),(9745,1114),(9749,2607)]
    ],
    "140g_rn16_rg1":[
        [(2319,2120),(2156,2745),(2296,2761),(2420,2255),(2343,2049),(2319,2120)],
        [(2311,3558),(2644,3595),(2624,3718),(2304,3690),(2311,3558)]
    ],
    "140g_rn16_rg2":[
        [(2388,9966),(2248,10196),(2359,10415),(3120,10209),(3106,8901),(2786,8473),(2340,9038),(2432,9820),(2388,9966)]
    ],
    "140g_rn16_rg3":[
        [(10388,7685),(10302,8018),(10375,8031),(10494,7714),(10494,7609),(10388,7685)]
    ],
    "140g_rn17_rg0":[
        [(11192,3696),(13648,2309),(14084,2852),(13603,5506),(12274,5043),(11191,4566),(11192,3696)]
    ],
    "140g_rn17_rg1":[
        [(6464,294),(6295,591),(6567,788),(6755,812),(6817,805),(6864,650),(6914,391),(6745,360),(6675,225),(6464,294)],
        [(5965,1584),(6143,1584),(6446,1774),(6654,1881),(6598,2019),(6223,1905),(5974,1767),(5883,1584),(5965,1584)],
        [(6701,1336),(6949,1194),(6928,1443),(6778,1526),(6687,1546),(6650,1498)],
        [(6786,2605),(6172,2881),(6209,3188),(6813,3095),(6922,2570),(6815,2415),(6786,2605)],
        [(6980,2922),(6885,3964),(6920,4329),(6941,4484),(7052,4478),(6988,4198),(7137,3991),(8027,3688),(8062,2346),(7487,1743),(7526,2564),(6908,2922)],
        [(7514,1188),(7520,1488),(7648,1432),(7629,1160),(7514,1188)],
        [(9062,1746),(9316,1688),(9310,1801),(9217,1843),(9058,1877),(9000,1815),(9062,1746)]
    ],
    "140g_rn17_rg2":[
        [(1682,3394),(1525,3914),(1502,4296),(1476,4539),(1495,4613),(1573,4610),(1612,4308),(1761,3917),(1931,3187),(1757,3252)],
        [(1684,1497),(1809,1423),(1878,1420),(1861,1489),(1784,1560),(1686,1604),(1627,1580)],
        [(2198,3145),(2517,3746),(2655,3914),(2586,3352),(2898,2929),(2831,2488),(2469,2610),(2454,3056),(2343,3154),(2291,3130),(2230,3101)],
        [(4090,2361),(4302,2352),(4277,2456),(4114,2462),(4026,2394),(4090,2361)]
    ],
    "140g_rn17_rg3":[
        [(3607,8828),(3494,9156),(3411,9435),(3359,9630),(3364,9763),(3429,9687),(3638,9149),(3684,8725),(3607,8828)]
    ],
    "140g_rn17_rg4":[
        [(7923,8075),(8123,8022),(8115,8105),(7934,8191),(7875,8144)],
        [(9883,7527),(10084,7414),(10140,7525),(9985,7682),(9834,7635),(9831,7486)],
        [(9636,8130),(9875,8548),(9934,8470),(9699,7998),(9600,8017)],
        [(8117,8216),(8146,8271),(8231,8266),(8234,8197),(8170,8175),(8117,8216)],
        [(9025,8376),(8909,8769),(8854,9178),(8848,9427),(8916,9400),(9038,8733),(9126,8340),(9025,8376)]
    ],
    "140g_rn18_rg0":[
        []
    ],
    "140g_rn18_rg1":[
        [(6568,9140),(6740,9519),(6847,9315),(6648,8973),(6568,9140)]
    ],
    "140g_rn18_rg2":[
        [(2323,8755),(1321,9793),(1363,10507),(1896,11162),(2286,11236),(2280,11641),(2453,11641),(2696,10927),(3108,10519),(3119,9594),(2589,8972),(2467,8612),(2323,8755)]
    ],
    "140g_rn18_rg3":[
        [(487,2791),(777,3952),(1498,3900),(1716,4149),(2313,3623),(2768,3007),(2859,2819),(3046,2856),(3120,2871),(3179,2881),(3286,2853),(2441,1754),(1938,1264),(1622,774),(649,1766),(487,2791)]
    ],
    "140g_rn18_rg4":[
        [(6618,1335),(6030,2500),(6519,3376),(7017,3244),(7270,2794),(7160,2122),(6912,1982),(6730,1587),(6694,1179),(6618,1335)]
    ],
    "140g_rn18_rg5":[
        []
    ],
    "140g_rn20_rg0":None,
    "140g_rn20_rg1":None,
    "140g_rn20_rg2":None,
    "140g_rn20_rg3":None,
    "140g_rn20_rg4":None,
    "140g_rn20_rg5":None,

    "140g_rn23_rg0":[[(4546,6),(4884,1633),(5324,2109),(6118,2159),(8110,1414),(8806,565),(8398,-83),(5082,-108),(4546,6)]],
    "140g_rn23_rg1":[[(2108,3108),(1918,5033),(2946,5771),(3034,3389),(2721,2480),(2108,3108)]],
    "140g_rn23_rg2":[
        [(5882,4963),(5059,5634),(5597,6026),(7743,5631),(7930,4876),(7311,5106),(6410,4863),(5882,4963)],
        [(5693,3152),(6781,3292),(8472,3505),(8149,4649),(6672,4066),(6208,4798),(5586,4834),(5529,3188)]
    ],
    "140g_rn23_rg3":[
        [(12612,4747),(13186,2985),(14276,3789),(13671,4401),(13359,5314),(12447,5511),(12612,4747)],
    ],
    "140g_rn23_rg4":[[]],
    "140g_rn23_rg5":[[]],
    "140g_rn25_rg0":[
        [(1130,562),(3663,1726),(3604,2286),(3123,4289),(1488,4113),(631,3435),(675,1155),(1130,562)]
    ],
    "140g_rn25_rg1":[
        [(7205,3787),(7600,4821),(7871,4897),(7603,4017),(7341,3598),(7205,3787)]
    ],
    "140g_rn25_rg2":[[]],
    "140g_rn25_rg3":[[]],
    "140g_rn25_rg4":[[
        (7723,11065),(7959,11498),(8023,11344),(7804,10905),(7723,11065)
        ]
    ],
    "140g_rn25_rg5":[
        [
            (974,8351),(2268,8642),(2924,9062),(3434,8957),(3350,9194),(2662,9874),(2893,11282),(1058,11025),(724,10713),(974,8351)
        ]
    ],
    "140g_rn26_rg0":[
        [(-30,6918),(522,6124),(1813,6928),(2093,7114),(2217,7049),(3249,6838),(3322,6965),(3223,7030),(3057,7092),(2742,8329),(2759,9235),(2291,9453),(1528,9319),(1134,8707),(393,9100),(77,7521)],
        [(2994,9057),(3076,8962),(3156,8882),(3227,8948),(3264,9202),(2957,9122),(2994,9057)],
        [(1991,9824),(2039,9999),(2123,10115),(2330,10057),(2521,9992),(2600,9770),(2740,9610),(2326,9690),(2080,9631),(1991,9824)]
    ],
    "140g_rn26_rg1":[
        [(6077,6954),(6804,6724),(7243,7295),(8131,7470),(9043,7310),(9139,7384),(8982,7466),(8292,8205),(8227,9125),(7772,9233),(6954,8802),(6203,8914),(5566,8765),(5916,7384),(6077,6954)],
        [(7785,9997),(7839,10198),(7900,10276),(8072,10324),(8283,10228),(8281,9868),(7785,9997)],
        [(8904,9363),(9070,9289),(9115,9418),(9092,9589),(8823,9418),(8904,9363)]
    ],
    "140g_rn26_rg2":[
        [(11608,9179),(12220,8908),(12557,8322),(12178,7745),(12644,7457),(12896,7608),(13063,7848),(13922,7814),(14002,8229),(13634,9467),(12946,9107),(11657,9553),(11608,9179)]
    ],
    "140g_rn26_rg3":[
        [(12062,2891),(12189,2835),(12337,2815),(12457,2797),(12458,2935),(11988,2996),(11987,2911),(12062,2891)]
    ],
    "140g_rn26_rg4":[
        [(4697,2262),(5092,2223),(5348,2266),(5237,2430),(4594,2328),(4592,2278),(4697,2262)]
    ],
}


DAPI_ALIGNMENTS = [ # Format: (y,x); downsampled image coords, # rotation is clockwise
    {
        "140g_rn3_rg0":{"center": (2540,3560),"z":7,"downsample_ratio":4,"rotation":0}, #0,# Rotation in image coordinate 
        "140g_rn3_rg1": {"center": (2970,3830), "z": 0, "downsample_ratio":4,"rotation":3},
        "140g_rn3_rg2": {"center": (3800,4600), "z": 0, "downsample_ratio":4,"rotation":-4},
        "140g_rn3_rg3": {"center": (4200,5000), "z": 0, "downsample_ratio":4,"rotation":-6}
    },
    {
        "140g_rn5_rg0": {"center": (3375,3175), "z":0, "downsample_ratio":4,"rotation":-12},
        "140g_rn5_rg1": {"center": (3950,4100), "z":0, "downsample_ratio":4,"rotation":-3},
        "140g_rn5_rg2": {"center": (4300,4700),"z":0, "downsample_ratio":4,"rotation":0},
        "140g_rn5_rg3": {"center": (4950,5550),"z":0, "downsample_ratio":4,"rotation":-8}
    },
    {
        # "140g_rn6_rg0": {"downsample_ratio":4,"z":0, "rotation":0},
        # "140g_rn6_rg1": {"center": (4840,4320), "downsample_ratio":4,"z": 7, "rotation":6},
        # "140g_rn6_rg2": {"center":(5632,4767), "downsample_ratio":4,"z":0, "rotation":-3},
        # "140g_rn6_rg3": {"downsample_ratio":4,"z":0, "rotation":0},
        "140g_rn6_rg4": {"center":(4355,3212),"downsample_ratio":4,"z":0, "rotation":0},
        # "140g_rn6_rg5": {"center": (3175,4180),"downsample_ratio":4,"z":0, "rotation":0},
    },
    {
        "140g_rn8_rg0": {"center": (6087,6239), "z":0, "downsample_ratio":4,"rotation":-9},
        "140g_rn8_rg1": {"center": (6189,6063), "z":5, "downsample_ratio":4,"rotation":-8.8},
        "140g_rn8_rg2": {"center": (5193,5680),"z":0, "downsample_ratio":4,"rotation":0},
        "140g_rn8_rg3": {"center": (5100,5440),"z":0, "downsample_ratio":4,"rotation":0},
        "140g_rn8_rg4": {"center": (4270,4000),"z":0, "downsample_ratio":4,"rotation":1}
    },
    {
        "140g_rn9_rg0": {"center": (4834,5895), "z":0, "downsample_ratio":4,"rotation":4.5},
        "140g_rn9_rg1": {"center": (4407,5096), "z":0, "downsample_ratio":4,"rotation":0},
        "140g_rn9_rg2": {"center": (4040,4920),"z":0, "downsample_ratio":4,"rotation":0},
        "140g_rn9_rg3": {"center": (2711,3992),"z":0, "downsample_ratio":4,"rotation":-2.5},
        "140g_rn9_rg4": {"center": (3350,5340),"z":0, "downsample_ratio":4,"rotation":-9.5},
        # "140g_rn9_rg5": {"center": (2318,2670),"z":0, "downsample_ratio":4,"rotation":0} # IGNORE THIS REGION (TOO SPARSE AND HARD TO ALIGN)
    },
    {
        "140g_rn7_rg0": {"center": (4415,4032), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn7_rg1": {"center": (4475,5085), "z":5, "downsample_ratio":4,"rotation":3.0},
        "140g_rn7_rg2": {"center": (3517,2192),"z":5, "downsample_ratio":4,"rotation":1},
        "140g_rn7_rg3": {"center": (4942,5348),"z":5, "downsample_ratio":4,"rotation":2.3},
        "140g_rn7_rg4": {"center": (5419,5390),"z":5, "downsample_ratio":4,"rotation":-2},
    },
    {
        "140g_rn10_rg0": {"center": (5710,5184), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn10_rg1": {"center": (5660,5250), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn10_rg2": {"center": (3552,3444),"z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn10_rg3": {"center": (3183,2572),"z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn10_rg4": {"center": (4820,5203),"z":5, "downsample_ratio":4,"rotation":0},
    },
    {
        "140g_rn11_rg0": {"center": (5020,4160), "z":5, "downsample_ratio":4,"rotation":-5},
        "140g_rn11_rg1": {"center": (5330,3710), "z":5, "downsample_ratio":4,"rotation":0},
        # "140g_rn11_rg2": {"center": (5020,4160), "z":5, "downsample_ratio":4,"rotation":-5},
        # "140g_rn11_rg3": {"center": (2685,1267), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn11_rg4": {"center": (3988,3232),"z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn11_rg5": {"center": (5880,4100),"z":5, "downsample_ratio":4,"rotation":0},
    },
    {
        "140g_rn12_rg0": {"center": (4110,2680), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn12_rg1": {"center": (3650,2130), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn12_rg2": {"center": (3147,2216), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn12_rg3": {"center": (3518,1531), "z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn12_rg4": {"center": (3460,2850),"z":5, "downsample_ratio":4,"rotation":0},
        "140g_rn12_rg5": {"center": (4690,2380),"z":5, "downsample_ratio":4,"rotation":0},
    },
    {
        #"140g_rn14_rg0": {"center": (0,4699), "z":7, "downsample_ratio":4,"rotation":-3},only half imaged can't align
        "140g_rn14_rg1": {"center": (4312,5565), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn14_rg2": {"center": (4414,5843), "z":7, "downsample_ratio":4,"rotation":3.5},
        "140g_rn14_rg3": {"center": (4250,4200), "z":7, "downsample_ratio":4,"rotation":4.5},
        "140g_rn14_rg4": {"center": (5680,5630),"z":7, "downsample_ratio":4,"rotation":6},
    },
    #10
    {
        "140g_rn15_rg0": {"center": (5380,3630), "z":7, "downsample_ratio":4,"rotation":6}, #10
        "140g_rn15_rg1": {"center": (5650,4200), "z":7, "downsample_ratio":4,"rotation":7},
        "140g_rn15_rg2": {"center": (4910,2680), "z":7, "downsample_ratio":4,"rotation":4.75},
        "140g_rn15_rg3": {"center": (2800,2200), "z":7, "downsample_ratio":4,"rotation":3},
        "140g_rn15_rg4": {"center": (4750,3810),"z":7, "downsample_ratio":4,"rotation":3.5},
        "140g_rn15_rg5": {"center": (6110,4200),"z":7, "downsample_ratio":4,"rotation":8},
    },
    {
        "140g_rn16_rg0": {"center": (4650,5160), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn16_rg1": {"center": (6525,4145), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn16_rg2": {"center": (4550,4650), "z":7, "downsample_ratio":4,"rotation":-8},
        "140g_rn16_rg3": {"center": (4050,3500), "z":7, "downsample_ratio":4,"rotation":0},
    },
    {
        "140g_rn17_rg0": {"center": (5100,5480), "z":7, "downsample_ratio":4,"rotation":-12},
        "140g_rn17_rg1": {"center": (6220,5190), "z":7, "downsample_ratio":4,"rotation":8},
        "140g_rn17_rg2": {"center": (4820,4710), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn17_rg3": {"center": (3075,2800), "z":7, "downsample_ratio":4,"rotation":-6.5},
        "140g_rn17_rg4": {"center": (4020,4500), "z":7, "downsample_ratio":4,"rotation":0},
    },
    {
        "140g_rn18_rg0": {"center": (3296,2594), "z":7, "downsample_ratio":4,"rotation":0},#good
        "140g_rn18_rg1": {"center": (6460,3170), "z":7, "downsample_ratio":4,"rotation":4},
        "140g_rn18_rg2": {"center": (5290,3950), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn18_rg3": {"center": (5720,4290), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn18_rg4": {"center": (4750,4020), "z":7, "downsample_ratio":4,"rotation":-3.5},#ok
        "140g_rn18_rg5": {"center": (4005,2620), "z":7, "downsample_ratio":4,"rotation":0},#good
    },
    {
        "140g_rn23_rg0": {"center": (190,6250), "z":7, "downsample_ratio":4,"rotation":0},#good
        "140g_rn23_rg1": {"center": (6040,3440), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn23_rg2": {"center": (4950,5790), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn23_rg3": {"center": (4820,5830), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn23_rg4": {"center": (3510,4130), "z":7, "downsample_ratio":4,"rotation":0},#ok
        "140g_rn23_rg5": {"center": (4100,4850), "z":7, "downsample_ratio":4,"rotation":0},#good
    },
    {
        "140g_rn25_rg0": {"center": (5840,4830), "z":7, "downsample_ratio":4,"rotation":0},#good
        "140g_rn25_rg1": {"center": (5130,3970), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn25_rg2": {"center": (3337,3085), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn25_rg3": {"center": (3426,3038), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn25_rg4": {"center": (4960,3660), "z":7, "downsample_ratio":4,"rotation":0},#ok
        "140g_rn25_rg5": {"center": (5630,4590), "z":7, "downsample_ratio":4,"rotation":0},#good
    },
    {
        "140g_rn26_rg0": {"center": (5740,2910), "z":7, "downsample_ratio":4,"rotation":0},#good
        "140g_rn26_rg1": {"center": (6120,6210), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn26_rg2": {"center": (5100,5870), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn26_rg3": {"center": (3522,3566), "z":7, "downsample_ratio":4,"rotation":0},
        "140g_rn26_rg4": {"center": (3690,5160), "z":7, "downsample_ratio":4,"rotation":0},#ok
    },

    {"140g_rn4_rg1": {"center": (3050,3250), "z": 7, "downsample_ratio":4,"rotation":0}, # Good
    "140g_rn4_rg3": {"center": (3500,4100), "z": 7, "downsample_ratio":4,"rotation":0},
    "140g_rn4_rg0":{"center": (3800,3900),"z":7,"downsample_ratio":4,"rotation":-2},
    "140g_rn4_rg2": {"center": (4600,4500), "z": 7, "downsample_ratio":4,"rotation":3},
    },
    

]

CELLBOUND2_ALIGNMENTS = [ # Format: (y,x) downsamples image coords
     # Rotation in image coordinate 
    {"140g_rn4_rg1": {"center": (3200,3100), "z": 0, "downsample_ratio":4,"rotation":-1}, # Ok
    "140g_rn4_rg3": {"center": (3500,4100), "z": 0, "downsample_ratio":4,"rotation":0},
    "140g_rn4_rg0":{"center": (3800,3900),"z":7,"downsample_ratio":4,"rotation":-2},
    "140g_rn4_rg2": {"center": (4600,4500), "z": 0, "downsample_ratio":4,"rotation":3},
    },
    {"140g_rn4_rg1": {"center": (3050,3250), "z": 0, "downsample_ratio":4,"rotation":0}, # Good
    "140g_rn4_rg3": {"center": (3500,4100), "z": 0, "downsample_ratio":4,"rotation":0},
    "140g_rn4_rg0":{"center": (3800,3900),"z":7,"downsample_ratio":4,"rotation":-2},
    "140g_rn4_rg2": {"center": (4600,4500), "z": 0, "downsample_ratio":4,"rotation":3},
    },
    {"140g_rn4_rg1": {"center": (3000,3250), "z": 0, "downsample_ratio":4,"rotation":3}, # Also good
    "140g_rn4_rg3": {"center": (3500,4100), "z": 0, "downsample_ratio":4,"rotation":0},
    "140g_rn4_rg0":{"center": (3800,3900),"z":7,"downsample_ratio":4,"rotation":-2},
    "140g_rn4_rg2": {"center": (4600,4500), "z": 0, "downsample_ratio":4,"rotation":3},
    },
]
