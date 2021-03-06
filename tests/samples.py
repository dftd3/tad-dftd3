# This file is part of tad-dftd3.
# SPDX-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from tad_dftd3 import util

structures = {
    "PbH4-BiH3": dict(
        numbers=util.to_number("Pb H H H H Bi H H H".split()),
        positions=torch.Tensor(
            [
                [-0.00000020988889, -4.98043478877778, +0.00000000000000],
                [+3.06964045311111, -6.06324400177778, +0.00000000000000],
                [-1.53482054188889, -6.06324400177778, -2.65838526500000],
                [-1.53482054188889, -6.06324400177778, +2.65838526500000],
                [-0.00000020988889, -1.72196703577778, +0.00000000000000],
                [-0.00000020988889, +4.77334244722222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, -2.35039772300000],
                [-2.71400388988889, +6.70626379422222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, +2.35039772300000],
            ]
        ),
        cn=torch.Tensor(
            [
                3.9388208389,
                0.9832025766,
                0.9832026958,
                0.9832026958,
                0.9865897894,
                2.9714603424,
                0.9870455265,
                0.9870456457,
                0.9870455265,
            ],
        ),
        weights=torch.tensor(
            [
                [1.107063e-27, 2.265990e-16, 1.057252e-07, 1.411678e-02, 9.858831e-01],
                [9.790892e-01, 2.091082e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.790892e-01, 2.091080e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.790892e-01, 2.091080e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.795891e-01, 2.041091e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [4.515681e-16, 1.310995e-07, 1.719178e-02, 9.828081e-01, 0.000000e-00],
                [9.796555e-01, 2.034455e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.796555e-01, 2.034453e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.796555e-01, 2.034455e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
            ],
        ),
        c6=torch.Tensor(
            [  # fmt: off
                [
                     456.209029,  37.274221,  37.274220,
                      37.274220,  37.263308, 493.665091,
                      37.261861,  37.261859,  37.261861,
                ],
                [
                      37.274221,   3.098765,   3.098765,
                       3.098765,   3.097897,  40.258369,
                       3.097782,   3.097782,   3.097782,
                ],
                [
                      37.274220,   3.098765,   3.098765,
                       3.098765,   3.097897,  40.258368,
                       3.097782,   3.097782,   3.097782,
                ],
                [
                      37.274220,   3.098765,   3.098765,
                       3.098765,   3.097897,  40.258368,
                       3.097782,   3.097782,   3.097782,
                ],
                [
                      37.263308,   3.097897,   3.097897,
                       3.097897,   3.097030,  40.246487,
                       3.096915,   3.096915,   3.096915,
                ],
                [
                     493.665091,  40.258369,  40.258368,
                      40.258368,  40.246487, 534.419974,
                      40.244911,  40.244910,  40.244911,
                ],
                [
                      37.261861,   3.097782,   3.097782,
                       3.097782,   3.096915,  40.244911,
                       3.096800,   3.096800,   3.096800,
                ],
                [
                      37.261859,   3.097782,   3.097782,
                       3.097782,   3.096915,  40.244910,
                       3.096800,   3.096800,   3.096800,
                ],
                [
                      37.261861,   3.097782,   3.097782,
                       3.097782,   3.096915,  40.244911,
                       3.096800,   3.096800,   3.096800,
                ],
            ],  # fmt: on
        ),
    ),
    "C6H5I-CH3SH": dict(
        numbers=util.to_number("C C C C C C I H H H H H S H C H H H".split(" ")),
        positions=torch.Tensor(
            [
                [-1.42754169820131, -1.50508961850828, -1.93430551124333],
                [+1.19860572924150, -1.66299114873979, -2.03189643761298],
                [+2.65876001301880, +0.37736955363609, -1.23426391650599],
                [+1.50963368042358, +2.57230374419743, -0.34128058818180],
                [-1.12092277855371, +2.71045691257517, -0.25246348639234],
                [-2.60071517756218, +0.67879949508239, -1.04550707592673],
                [-2.86169588073340, +5.99660765711210, +1.08394899986031],
                [+2.09930989272956, -3.36144811062374, -2.72237695164263],
                [+2.64405246349916, +4.15317840474646, +0.27856972788526],
                [+4.69864865613751, +0.26922271535391, -1.30274048619151],
                [-4.63786461351839, +0.79856258572808, -0.96906659938432],
                [-2.57447518692275, -3.08132039046931, -2.54875517521577],
                [-5.88211879210329, 11.88491819358157, +2.31866455902233],
                [-8.18022701418703, 10.95619984550779, +1.83940856333092],
                [-5.08172874482867, 12.66714386256482, -0.92419491629867],
                [-3.18311711399702, 13.44626574330220, -0.86977613647871],
                [-5.07177399637298, 10.99164969235585, -2.10739192258756],
                [-6.35955320518616, 14.08073002965080, -1.68204314084441],
            ]
        ),
        cn=torch.Tensor(
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ),
        weights=torch.Tensor(
            [
                [7.669145e-18, 9.045589e-09, 5.554739e-03, 9.361996e-01, 5.824569e-02],
                [9.333375e-18, 1.033050e-08, 5.943523e-03, 9.392180e-01, 5.483844e-02],
                [7.667672e-18, 9.044414e-09, 5.554371e-03, 9.361965e-01, 5.824912e-02],
                [9.612786e-20, 4.548463e-10, 1.160488e-03, 7.995832e-01, 1.992563e-01],
                [7.996743e-18, 9.305203e-09, 5.635430e-03, 9.368604e-01, 5.750420e-02],
                [9.744076e-20, 4.591300e-10, 1.166394e-03, 8.002455e-01, 1.985881e-01],
                [2.542048e-04, 9.997458e-01, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.819180e-01, 1.808196e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.830121e-01, 1.698789e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.819358e-01, 1.806416e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.830075e-01, 1.699249e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.819358e-01, 1.806418e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.188072e-09, 5.005845e-03, 9.949941e-01, 0.000000e-00, 0.000000e-00],
                [9.812128e-01, 1.878718e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [2.610153e-28, 2.424125e-16, 1.386950e-07, 2.015082e-02, 9.798490e-01],
                [9.833007e-01, 1.669934e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.830270e-01, 1.697298e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
                [9.825624e-01, 1.743759e-02, 0.000000e-00, 0.000000e-00, 0.000000e-00],
            ],
        ),
        c6=torch.Tensor(
            [  # fmt: off
                [
                   25.308671,  25.323615,  25.308655,  24.710076,  25.311916,  24.712888,
                   94.892296,   8.828577,   8.823193,   8.828489,   8.823216,   8.828490,
                   56.245532,   8.832048,  21.436007,   8.821773,   8.823120,   8.825406,
                ],
                [
                   25.323615,  25.338571,  25.323599,  24.724560,  25.326863,  24.727374,
                   94.950852,   8.833831,   8.828443,   8.833743,   8.828466,   8.833744,
                   56.280052,   8.837304,  21.447970,   8.827023,   8.828370,   8.830658,
                ],
                [
                   25.308655,  25.323599,  25.308639,  24.710061,  25.311900,  24.712873,
                   94.892234,   8.828571,   8.823187,   8.828484,   8.823210,   8.828484,
                   56.245495,   8.832042,  21.435994,   8.821768,   8.823114,   8.825401,
                ],
                [
                   24.710076,  24.724560,  24.710061,  24.130008,  24.713222,  24.132733,
                   92.545041,   8.618185,   8.612959,   8.618100,   8.612981,   8.618100,
                   54.862146,   8.621554,  20.957466,   8.611582,   8.612888,   8.615108,
                ],
                [
                   25.311916,  25.326863,  25.311900,  24.713222,  25.315162,  24.716034,
                   94.905014,   8.829718,   8.824333,   8.829630,   8.824356,   8.829631,
                   56.253029,   8.833189,  21.438605,   8.822913,   8.824260,   8.826547,
                ],
                [
                   24.712888,  24.727374,  24.712873,  24.132733,  24.716034,  24.135458,
                   92.556069,   8.619173,   8.613947,   8.619088,   8.613969,   8.619089,
                   54.868645,   8.622542,  20.959714,   8.612569,   8.613876,   8.616096,
                ],
                [
                   94.892296,  94.950852,  94.892234,  92.545041,  94.905014,  92.556069,
                  358.497837,  33.161012,  33.140079,  33.160672,  33.140167,  33.160673,
                  212.283941,  33.174507,  79.702741,  33.134559,  33.139793,  33.148685,
                ],
                [
                    8.828577,   8.833831,   8.828571,   8.618185,   8.829718,   8.619173,
                   33.161012,   3.088957,   3.087062,   3.088926,   3.087070,   3.088926,
                   19.679250,   3.090178,   7.467556,   3.086563,   3.087036,   3.087841,
                ],
                [
                    8.823193,   8.828443,   8.823187,   8.612959,   8.824333,   8.613947,
                   33.140079,   3.087062,   3.085169,   3.087031,   3.085177,   3.087032,
                   19.666885,   3.088283,   7.463197,   3.084670,   3.085143,   3.085947,
                ],
                [
                    8.828489,   8.833743,   8.828484,   8.618100,   8.829630,   8.619088,
                   33.160672,   3.088926,   3.087031,   3.088895,   3.087039,   3.088895,
                   19.679050,   3.090148,   7.467485,   3.086532,   3.087006,   3.087810,
                ],
                [
                    8.823216,   8.828466,   8.823210,   8.612981,   8.824356,   8.613969,
                   33.140167,   3.087070,   3.085177,   3.087039,   3.085185,   3.087040,
                   19.666937,   3.088291,   7.463216,   3.084678,   3.085151,   3.085955,
                ],
                [
                    8.828490,   8.833744,   8.828484,   8.618100,   8.829631,   8.619089,
                   33.160673,   3.088926,   3.087032,   3.088895,   3.087040,   3.088896,
                   19.679050,   3.090148,   7.467485,   3.086532,   3.087006,   3.087811,
                ],
                [
                   56.245532,  56.280052,  56.245495,  54.862146,  56.253029,  54.868645,
                  212.283941,  19.679250,  19.666885,  19.679050,  19.666937,  19.679050,
                  125.836574,  19.687222,  47.294189,  19.663624,  19.666716,  19.671969,
                ],
                [
                    8.832048,   8.837304,   8.832042,   8.621554,   8.833189,   8.622542,
                   33.174507,   3.090178,   3.088283,   3.090148,   3.088291,   3.090148,
                   19.687222,   3.091400,   7.470366,   3.087783,   3.088257,   3.089062,
                ],
                [
                   21.436007,  21.447970,  21.435994,  20.957466,  21.438605,  20.959714,
                   79.702741,   7.467556,   7.463197,   7.467485,   7.463216,   7.467485,
                   47.294189,   7.470366,  18.341314,   7.462048,   7.463138,   7.464989,
                ],
                [
                    8.821773,   8.827023,   8.821768,   8.611582,   8.822913,   8.612569,
                   33.134559,   3.086563,   3.084670,   3.086532,   3.084678,   3.086532,
                   19.663624,   3.087783,   7.462048,   3.084171,   3.084644,   3.085448,
                ],
                [
                    8.823120,   8.828370,   8.823114,   8.612888,   8.824260,   8.613876,
                   33.139793,   3.087036,   3.085143,   3.087006,   3.085151,   3.087006,
                   19.666716,   3.088257,   7.463138,   3.084644,   3.085117,   3.085921,
                ],
                [
                    8.825406,   8.830658,   8.825401,   8.615108,   8.826547,   8.616096,
                   33.148685,   3.087841,   3.085947,   3.087810,   3.085955,   3.087811,
                   19.671969,   3.089062,   7.464989,   3.085448,   3.085921,   3.086726,
                ],
            ],  # fmt: on
        ),
    ),
    "C4H5NCS": dict(
        numbers=util.to_number(symbols="C C C C N C S H H H H H".split()),
        positions=torch.Tensor(
            [
                [-2.56745685564671, -0.02509985979910, 0.00000000000000],
                [-1.39177582455797, +2.27696188880014, 0.00000000000000],
                [+1.27784995624894, +2.45107479759386, 0.00000000000000],
                [+2.62801937615793, +0.25927727028120, 0.00000000000000],
                [+1.41097033661123, -1.99890996077412, 0.00000000000000],
                [-1.17186102298849, -2.34220576284180, 0.00000000000000],
                [-2.39505990368378, -5.22635838332362, 0.00000000000000],
                [+2.41961980455457, -3.62158019253045, 0.00000000000000],
                [-2.51744374846065, +3.98181713686746, 0.00000000000000],
                [+2.24269048384775, +4.24389473203647, 0.00000000000000],
                [+4.66488984573956, +0.17907568006409, 0.00000000000000],
                [-4.60044244782237, -0.17794734637413, 0.00000000000000],
            ]
        ),
    ),
}
