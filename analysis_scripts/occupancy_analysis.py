
import uproot
import numpy as np 
import hist
import math
import mplhep as hep
import xml.etree.ElementTree as ET
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
from collections import Counter,defaultdict
import traceback

from src import detector_config

from src.hit_analysis.plotting import analyze_detectors_and_plot
from src.geometry_parsing.k4geo_parsers import parse_detector_constants
from src.geometry_parsing.geometry_info import get_geometry_info

from src.detector_config import get_detector_configs, get_xmls

from src.hit_analysis.train_analyzer import analyze_detectors_and_plot_by_train

# Import parallel I/O utilities
from src.utils.parallel_io import (
    open_files_parallel, 
    open_files_in_chunks,
    validate_file_paths,
    estimate_memory_usage
)
from src.utils.parallel_config import get_optimal_worker_count




# List of seeds
#seeds = [527875,733579,360177,286038,479728]  # Replace with your actual list of seeds

# # First batch of seeds - 60,
# seeds_batch_1 = [
#     646726, 824897, 685736, 268021, 893161, 473632, 570691, 973731, 172660, 588225, 
#     834692, 765514, 387394, 382466, 893505, 682500, 107626, 581802, 223639, 907367, 
#     168136, 116898, 878488, 974354, 429435, 514355, 595020, 298860, 318192, 795000, 
#     516831, 938118, 219951, 473269, 620860, 549738, 201099, 969835, 328981, 649096, 
#     600364, 495079, 860906, 277515, 883098, 789080, 208830, 751176, 826856, 521374, 
#     446610, 442571, 977237, 956193, 638935, 629828, 946409, 711382, 838586, 478805
# ]

# # Second batch of seeds - 73
# seeds_batch_2 = [
#     568241, 361733, 718355, 670236, 392403, 523638, 775506, 619237, 887397, 796413, 
#     307313, 932023, 665869, 180940, 981910, 992834, 541986, 635763, 858530, 361884, 
#     501895, 527488, 561072, 982265, 586359, 982832, 743694, 330631, 679469, 558889, 
#     749209, 573285, 520380, 994774, 619288, 307004, 368504, 910818, 138965, 443531, 
#     928724, 947691, 901624, 393914, 177378, 321364, 599380, 739432, 966089, 815377, 
#     733980, 797286, 784389, 807508, 291957, 140752, 800660, 736089, 183876, 409570, 
#     779944, 544875, 939767, 428507, 147203, 734028, 511643, 671095, 561651, 774906, 
#     724329, 631167, 596798
# ]

# # Third batch of seeds - 60
# seeds_batch_3 = [
#     527875, 733579, 360177, 286038, 479728, 750622, 869004, 616205, 664707, 132661, 
#     781198, 426737, 328457, 144480, 621961, 768697, 685339, 419765, 809804, 564915, 
#     262728, 985882, 256863, 353443, 409948, 725103, 996117, 115337, 182286, 909242, 
#     984689, 999153, 696557, 949647, 890767, 228995, 677476, 584121, 730711, 321485, 
#     446871, 447608, 771760, 190219, 128353, 536212, 886981, 711233, 790240, 577473, 
#     524857, 886026, 376225, 440800, 831544, 541993, 576716, 106733, 876089, 754625
# ]

# # Fourth batch of seeds - 73
# seeds_batch_4 = [
#     340252, 566577, 438336, 983321, 362111, 175226, 711267, 185912, 312192, 847048, 
#     400945, 561096, 271634, 831467, 711631, 875591, 203540, 473757, 226527, 595212, 
#     917314, 256322, 434222, 840790, 656615, 150344, 255367, 164750, 713978, 431949, 
#     804069, 481013, 901407, 342973, 476156, 736050, 469163, 457657, 934359, 664781, 
#     428914, 998147, 265876, 614237, 897014, 939699, 367571, 894524, 133257, 323854, 
#     984156, 759221, 779288, 124405, 474231, 183930, 232355, 597546, 426705, 311767, 
#     919859, 646517, 471934, 880243, 142979, 919932, 659688, 928661, 455310, 975581, 
#     506428, 745883, 539398
# ]

#seeds = [seeds_batch_1,seeds_batch_2,seeds_batch_3,seeds_batch_4]

#seeds_test = [seeds_batch_1]
#seeds = [seeds_batch_1]
#seeds = [seeds_test]


#seeds = [seeds_batch_1,seeds_batch_2]
#seeds = [seeds_batch_1]


seeds_batch_1 = [
   106733, 107626, 112380, 113032, 115337, 116898, 118134, 120286, 124228, 124405,
   128353, 129910, 131674, 132661, 133257, 138304, 138965, 140752, 141286, 142979,
   144480, 145443, 145627, 147203, 150344, 151725, 152849, 154253, 159477, 164308,
   164430, 164473, 164710, 164750, 167072, 167330, 168136, 169124, 171098, 172660,
   174526, 175226, 175647, 176035, 176672, 177378, 178466, 180940, 181031, 182286,
   183876, 183930, 183931, 184449, 185707, 185912, 189967, 190219, 193713, 195324,
   201099, 203540, 208830, 209556, 211793, 215801, 216183, 216941, 217735, 219310,
   219951, 220937, 221632, 223619, 223639, 226527, 228695, 228995, 230620, 231898,
   232355, 235906, 236678, 237410, 238485, 240253, 244071, 246388, 248279, 250244,
   252370, 255362, 255367, 255768, 256147, 256322, 256863, 256940, 257575, 257990,
   261291, 261905, 262420, 262728, 262834, 264875, 265876, 265941, 268021, 268250,
   269021, 269897, 271634, 272889, 277515, 278034, 278521, 281257, 281875, 282266,
   286038, 287031, 288880, 289454, 290592, 290881, 291957, 294038, 295469, 298860
]

seeds_batch_2 = [
   302620, 302985, 303154, 304185, 304775, 304820, 305987, 307004, 307313, 311767,
   312192, 313580, 316100, 316297, 318192, 318599, 318749, 320841, 321364, 321485,
   322196, 322471, 323854, 326172, 326248, 326431, 327874, 328457, 328981, 330631,
   333532, 334450, 334830, 337793, 338465, 338817, 340143, 340252, 340901, 342973,
   347376, 353443, 356431, 358480, 358518, 360070, 360177, 361287, 361622, 361733,
   361884, 361990, 362111, 362495, 363713, 367518, 367571, 367967, 368504, 369471,
   370773, 370892, 372641, 373353, 376225, 377175, 378490, 380205, 380514, 382466,
   383687, 384611, 386752, 386899, 387394, 387858, 388167, 388773, 391205, 391404,
   391491, 392113, 392403, 392614, 392630, 393914, 394247, 396180, 398469, 399007,
   400710, 400945, 401163, 403121, 403510, 403916, 404440, 406445, 406478, 406500,
   407436, 409570, 409650, 409948, 411417, 411768, 416299, 416553, 416562, 417284,
   417572, 419765, 420272, 421922, 422461, 423076, 423145, 424279, 424641, 426705,
   426737, 426793, 428507, 428677, 428914, 429435, 429546, 430155, 430857, 431949
]

seeds_batch_3 = [
   433719, 434222, 436364, 437029, 438336, 440800, 440869, 441487, 442571, 442595,
   442727, 443531, 445996, 446610, 446617, 446871, 447608, 449613, 450283, 450635,
   451315, 452402, 455310, 456022, 456696, 456711, 457657, 459452, 459510, 459609,
   460121, 460149, 460966, 461862, 462076, 462710, 467470, 469163, 469542, 470424,
   471934, 473269, 473632, 473757, 474231, 474751, 475914, 475933, 476156, 476925,
   478805, 479392, 479728, 479795, 480087, 480178, 481013, 481779, 481823, 482831,
   483279, 485829, 486351, 488285, 489164, 490942, 492895, 494702, 495079, 495847,
   496654, 497284, 499544, 499852, 500955, 501895, 502353, 502422, 502637, 503614,
   504455, 504649, 505212, 506259, 506428, 507673, 508174, 508220, 508314, 508516,
   509258, 509696, 510112, 510743, 510972, 511200, 511636, 511643, 513552, 514232,
   514355, 516128, 516447, 516831, 516979, 517053, 517750, 518093, 518350, 519054,
   519075, 520380, 520737, 521374, 521853, 522226, 523638, 524391, 524816, 524857,
   524896, 525318, 525372, 526665, 527412, 527488, 527651, 527875, 527901, 530305
]

seeds_batch_4 = [
   531239, 531584, 532013, 532237, 532442, 533105, 534447, 534727, 535191, 536212,
   537456, 538852, 539204, 539283, 539377, 539398, 539462, 541233, 541948, 541986,
   541993, 542329, 543233, 543329, 543862, 544875, 546021, 546917, 546978, 547476,
   548043, 548401, 549738, 550195, 551131, 551780, 552023, 552721, 554699, 555460,
   555669, 557029, 557269, 558708, 558752, 558889, 560618, 561072, 561096, 561651,
   561880, 562006, 563412, 563512, 563862, 564915, 564963, 566577, 567532, 568241,
   569731, 570223, 570545, 570691, 571023, 571307, 571327, 572186, 572848, 573234,
   573285, 574170, 576164, 576345, 576554, 576716, 577473, 577828, 578158, 578711,
   578978, 579319, 579704, 579737, 580298, 580785, 581802, 583314, 584121, 585006,
   585187, 585344, 585419, 585479, 586144, 586359, 588225, 588399, 589158, 589223,
   590216, 590598, 590732, 591472, 591921, 592184, 593265, 594010, 594595, 595020,
   595212, 595612, 596087, 596115, 596798, 596944, 597126, 597389, 597546, 598156,
   599380, 600364, 600586, 600612, 601351, 601532, 601567, 601782, 602042, 602460
]

seeds_batch_5 = [
   603035, 603072, 603181, 604340, 604617, 604907, 605273, 605342, 605481, 605930,
   606835, 607362, 607578, 607624, 608146, 609292, 609305, 609476, 609754, 610255,
   611346, 612711, 612810, 612907, 613090, 613908, 614119, 614237, 614567, 615512,
   615821, 616205, 616699, 616899, 616939, 617491, 618295, 618750, 619227, 619237,
   619288, 619602, 619727, 619832, 619870, 620006, 620790, 620860, 621375, 621961,
   622808, 622876, 623039, 623234, 623883, 625329, 625835, 626926, 627277, 627669,
   627905, 629828, 630698, 631167, 631412, 632054, 632723, 634885, 635071, 635469,
   635618, 635763, 635801, 636685, 636882, 636953, 637163, 637374, 637643, 637866,
   637929, 637955, 638067, 638935, 639466, 639964, 640537, 641294, 641380, 641865,
   642520, 642659, 643581, 643918, 644461, 644853, 645406, 645852, 646070, 646517,
   646726, 647351, 647505, 647894, 648320, 648410, 648479, 649096, 650136, 650288,
   650382, 652031, 652505, 652516, 653699, 654265, 654281, 654383, 655483, 655531,
   656615, 657124, 657508, 657969, 658330, 659688, 659865, 659940, 660424, 661090
]

seeds_batch_6 = [
   661443, 661746, 662076, 662577, 662774, 662851, 663395, 663574, 663711, 664246,
   664266, 664328, 664707, 664781, 665442, 665690, 665869, 665877, 665958, 666107,
   666177, 666180, 666287, 666646, 667138, 667167, 667986, 668180, 668628, 668970,
   668979, 669271, 669844, 670236, 670955, 671095, 671317, 672344, 673286, 673475,
   674882, 674955, 675523, 676038, 676483, 676525, 676761, 676982, 676996, 677476,
   677622, 678270, 678709, 679261, 679469, 679782, 679962, 680025, 680583, 680677,
   681507, 681836, 681917, 682311, 682500, 682921, 683222, 683317, 683677, 685339,
   685404, 685685, 685736, 687291, 687919, 688195, 688517, 688816, 688891, 689613,
   689970, 690376, 690962, 691216, 691570, 692059, 692101, 693005, 693036, 693387,
   693647, 693693, 693814, 694723, 694760, 694890, 695031, 695334, 695572, 695764,
   696182, 696557, 697101, 697768, 698622, 698814, 700087, 700431, 700865, 700910,
   701582, 701603, 701686, 702564, 702602, 703417, 703498, 703815, 704146, 704325,
   704592, 704690, 705655, 706954, 708310, 708340, 708356, 708462, 708821, 709101
]

seeds_batch_7 = [
   709381, 709586, 709669, 709734, 709900, 710320, 710385, 710423, 710545, 710775,
   711198, 711233, 711267, 711382, 711430, 711431, 711591, 711631, 711649, 711662,
   711753, 711866, 711974, 712210, 712250, 712647, 712652, 712738, 712891, 713407,
   713449, 713727, 713923, 713944, 713978, 714683, 714745, 714886, 715409, 715461,
   715464, 716034, 716416, 716725, 716863, 717423, 718174, 718342, 718355, 718371,
   718435, 718438, 718634, 718658, 718714, 718719, 718910, 719179, 719926, 720204,
   720349, 721954, 722166, 722326, 722478, 722949, 722952, 723019, 723520, 723522,
   724142, 724162, 724198, 724325, 724329, 724389, 724606, 724707, 724758, 724783,
   725003, 725103, 725456, 725762, 725914, 725953, 725989, 726229, 726593, 726908,
   726918, 727054, 727263, 727323, 728215, 728668, 728731, 728792, 728854, 729242,
   729424, 729847, 729985, 730062, 730339, 730389, 730703, 730711, 730876, 730989,
   731247, 731268, 731349, 731664, 731830, 732116, 732204, 732756, 732921, 732985,
   733352, 733403, 733473, 733578, 733579, 733650, 733961, 733980, 734027, 734028
]

seeds_batch_8 = [
   735093, 735117, 735298, 735360, 735632, 735860, 735914, 736024, 736050, 736089,
   736413, 737182, 737254, 737265, 737853, 737882, 738247, 738582, 738820, 739344,
   739432, 739835, 740023, 740051, 740076, 740151, 741029, 741284, 741638, 741710,
   741756, 741887, 742071, 742543, 742585, 743511, 743694, 743825, 743985, 744947,
   745094, 745132, 745272, 745883, 746483, 746787, 746820, 746983, 747899, 748127,
   748462, 748526, 748540, 748566, 748909, 749145, 749209, 749216, 749237, 750475,
   750622, 751079, 751084, 751176, 751372, 751631, 752507, 752941, 753277, 753532,
   753661, 753761, 754560, 754625, 754784, 755565, 756117, 756202, 757050, 757249,
   757442, 757592, 758680, 759044, 759213, 759221, 759591, 760175, 760348, 760350,
   760430, 760553, 760727, 760767, 760874, 761315, 761727, 761918, 762080, 762111,
   762140, 762492, 762564, 762888, 762893, 762894, 763047, 763055, 763117, 763130,
   763143, 763875, 763979, 764132, 764241, 764529, 764548, 765181, 765415, 765514,
   765651, 765788, 765789, 765796, 766312, 767142, 767171, 767373, 767637, 768179
]

seeds_batch_9 = [
   768290, 768448, 768500, 768572, 768662, 768697, 768782, 768831, 768869, 768881,
   769014, 769030, 769264, 769273, 769364, 769583, 769781, 769942, 770662, 771273,
   771411, 771760, 771892, 771924, 771965, 772180, 772240, 772547, 772563, 772889,
   773383, 773685, 773981, 774132, 774575, 774906, 774939, 775134, 775506, 775590,
   775613, 776142, 776441, 776514, 776664, 776840, 776891, 777101, 777197, 777224,
   777505, 778224, 778354, 778652, 778722, 778783, 779027, 779238, 779288, 779657,
   779938, 779944, 780450, 780541, 780621, 780786, 780911, 781198, 781299, 781414,
   781868, 782194, 782328, 782695, 782743, 782851, 783088, 783421, 783461, 783592,
   783916, 784389, 784518, 785086, 785213, 786370, 786400, 787036, 787109, 787235,
   787530, 788058, 788428, 788467, 788921, 788942, 789048, 789080, 789140, 790240,
   790263, 790706, 790923, 791047, 791450, 791621, 791631, 791706, 792023, 792415,
   792756, 792784, 793611, 793649, 793731, 794129, 794764, 794798, 794843, 795000,
   795045, 795301, 795465, 795557, 795688, 795834, 796302, 796335, 796376, 796413
]

seeds_batch_10 = [
   796496, 796978, 797110, 797156, 797286, 797333, 797369, 797387, 797530, 798440,
   798476, 798499, 798660, 798941, 798945, 799634, 800453, 800480, 800660, 800702,
   800899, 801003, 801316, 801462, 801599, 801856, 802073, 802158, 802239, 802262,
   802810, 802858, 802957, 803151, 803260, 803262, 803435, 803510, 803703, 803706,
   803981, 804069, 804075, 804153, 804241, 804244, 804411, 804443, 804653, 804807,
   805288, 805347, 805468, 805733, 805921, 805975, 806366, 806401, 806545, 806677,
   806698, 806786, 807098, 807171, 807292, 807455, 807508, 807888, 808041, 808378,
   808474, 808763, 809183, 809184, 809804, 809981, 810038, 810116, 810824, 810849,
   810989, 811574, 812034, 812117, 812343, 812484, 812781, 813297, 813333, 813384,
   813957, 814135, 814379, 814516, 814615, 814673, 814815, 814959, 815197, 815218,
   815377, 815456, 815909, 816670, 817094, 817188, 817639, 817911, 818013, 818116,
   818183, 818342, 818531, 818613, 818735, 819082, 819090, 819260, 819455, 819543,
   819760, 820254, 820559, 820625, 820708, 821166, 821230, 821615, 821703, 822013
]

seeds_batch_11 = [
   822148, 822243, 822246, 822762, 822818, 822839, 822947, 822958, 823003, 823041,
   823230, 823418, 823992, 824070, 824140, 824896, 824897, 825133, 825346, 825434,
   825554, 825649, 826629, 826725, 826856, 827157, 827167, 827353, 827387, 827440,
   827600, 827808, 828029, 828422, 828635, 828844, 829379, 829458, 829744, 829787,
   829834, 829972, 830211, 830564, 830651, 830743, 830791, 831050, 831467, 831544,
   831700, 831753, 831783, 831863, 831909, 832389, 832565, 833405, 833439, 833487,
   833769, 833791, 834221, 834363, 834671, 834692, 835314, 835684, 835769, 835976,
   836059, 836505, 837057, 837172, 837375, 837505, 837620, 837659, 837704, 837997,
   838586, 838703, 838846, 839296, 839552, 839653, 839739, 839790, 839916, 840790,
   840899, 841199, 841636, 841944, 842106, 842217, 842391, 842496, 842634, 842653,
   842739, 842870, 843546, 844464, 844571, 844625, 844720, 844823, 845110, 845199,
   845565, 845927, 846672, 846771, 846880, 847030, 847048, 847168, 847227, 847443,
   848138, 849549, 850179, 851183, 851521, 851689, 851782, 851883, 852740, 852770
]

seeds_batch_12 = [
   852985, 853140, 853231, 853346, 853357, 853606, 853843, 854052, 854140, 854425,
   854682, 855080, 855254, 856241, 856706, 857118, 857710, 857711, 857731, 858141,
   858342, 858364, 858530, 858771, 859006, 859007, 859027, 859163, 859423, 860631,
   860906, 861163, 861254, 861262, 861365, 861542, 862143, 862580, 862632, 862679,
   862747, 863164, 863491, 863565, 864333, 864383, 864414, 864515, 864590, 864611,
   864673, 866021, 866435, 866705, 867210, 867256, 867733, 868154, 868227, 868452,
   869004, 869252, 869417, 869470, 869541, 869885, 869995, 870250, 870431, 870458,
   870633, 871383, 871502, 871574, 871968, 872000, 872043, 872566, 872728, 873124,
   873198, 873464, 873629, 873776, 873906, 874361, 874660, 874765, 874779, 874925,
   875051, 875200, 875580, 875591, 875704, 876024, 876089, 876213, 876813, 876965,
   876974, 876990, 877001, 877682, 877760, 877767, 877818, 877905, 877935, 878488,
   878931, 879063, 879474, 879542, 879837, 879876, 879918, 879923, 880243, 880322,
   880389, 880672, 881024, 881166, 881200, 881345, 881621, 881699, 881922, 882316
]

seeds_batch_13 = [
   882343, 882558, 882861, 883098, 883301, 883359, 884816, 885006, 885070, 885527,
   885640, 885804, 886026, 886103, 886722, 886981, 887073, 887160, 887218, 887397,
   887523, 887562, 887567, 887717, 887959, 888126, 888340, 888489, 888496, 888678,
   888821, 888987, 889819, 889975, 890081, 890340, 890553, 890701, 890767, 890956,
   890957, 891043, 891729, 891956, 892564, 892619, 892674, 893078, 893161, 893373,
   893413, 893505, 893710, 893739, 894045, 894053, 894159, 894352, 894452, 894524,
   894722, 894995, 895150, 895680, 895684, 895818, 896194, 896395, 896629, 896893,
   897014, 897189, 897299, 897310, 897395, 898376, 899095, 900056, 900358, 900612,
   900730, 900828, 901282, 901326, 901407, 901624, 901631, 901985, 901996, 902128,
   902148, 902383, 902621, 902637, 902949, 902985, 904369, 904488, 904785, 905049,
   905208, 905446, 905572, 905698, 905822, 906299, 906767, 907367, 907927, 908041,
   908193, 908481, 909242, 909271, 909471, 909492, 909842, 909954, 910043, 910314,
   910357, 910818, 910902, 910971, 911150, 911171, 911196, 911216, 911234, 911325
]

seeds_batch_14 = [
   911828, 911903, 912333, 913146, 913159, 913210, 913724, 913875, 914489, 914808,
   914896, 914953, 915050, 915190, 915225, 915731, 915985, 916255, 916432, 916773,
   917165, 917242, 917314, 917537, 917798, 917908, 918186, 918810, 919080, 919669,
   919859, 919915, 919932, 919995, 920568, 920886, 921087, 921311, 921464, 921572,
   921728, 921851, 922080, 922410, 922517, 922552, 923931, 923972, 923989, 924472,
   925267, 925280, 925391, 926361, 926393, 926748, 926961, 927322, 927532, 927568,
   927919, 927938, 928508, 928650, 928661, 928724, 928934, 929098, 929281, 929548,
   929771, 929973, 930263, 930289, 930355, 930421, 930536, 930592, 931349, 931395,
   931769, 932023, 932041, 932161, 932226, 932303, 932319, 932344, 932606, 932962,
   933319, 934215, 934344, 934359, 934444, 934448, 934812, 935129, 935322, 935569,
   935828, 936027, 936646, 936664, 936944, 937325, 937378, 937532, 938118, 938364,
   938661, 938781, 938811, 938869, 939001, 939169, 939240, 939376, 939699, 939753,
   939767, 940361, 940380, 940477, 940530, 940822, 941254, 941729, 941807, 942205
]

seeds_batch_15 = [
   942428, 942612, 943205, 943232, 943353, 943400, 943721, 944295, 944848, 944855,
   944861, 944956, 945675, 946386, 946409, 947446, 947691, 948271, 948516, 948620,
   948887, 949089, 949154, 949647, 950108, 950604, 950716, 950923, 951076, 951440,
   951759, 952100, 952210, 952328, 952656, 952755, 952924, 952970, 953109, 953400,
   953439, 953466, 953472, 953497, 953539, 953568, 953630, 953840, 953923, 954181,
   954575, 954776, 954901, 955163, 955879, 955974, 956193, 956228, 956413, 956680,
   956878, 956959, 957053, 957535, 957986, 958186, 958438, 958629, 958808, 958893,
   958995, 959233, 959626, 960175, 960381, 960427, 961349, 961663, 961841, 961866,
   961990, 962066, 962140, 962183, 962218, 962463, 963040, 963408, 963568, 963595,
   963719, 963814, 964100, 964337, 964482, 964503, 964633, 965006, 965155, 965227,
   965310, 965342, 965584, 965797, 965862, 966089, 966149, 966505, 966713, 967090,
   967124, 967291, 967420, 967887, 968073, 968443, 968460, 969835, 970129, 970449,
   970571, 970636, 970697, 971029, 971481, 971814, 971816, 972386, 972711, 972887
]

seeds_batch_16 = [
   973335, 973731, 973860, 974060, 974186, 974354, 974571, 974600, 974706, 974767,
   974831, 974879, 974888, 975094, 975299, 975581, 975790, 976565, 976755, 976956,
   977001, 977128, 977237, 977276, 978413, 978705, 978842, 979032, 979042, 979193,
   979284, 979514, 979534, 979597, 979681, 980078, 980082, 980750, 981184, 981223,
   981910, 982265, 982832, 983163, 983321, 983625, 984139, 984156, 984610, 984684,
   984689, 984782, 985080, 985165, 985217, 985438, 985882, 986243, 986434, 986830,
   987065, 987567, 987597, 987897, 988010, 988342, 988439, 988463, 988859, 988920,
   989145, 989598, 989875, 989967, 989971, 990036, 990085, 990266, 990844, 991399,
   991717, 991819, 992294, 992400, 992735, 992788, 992834, 993307, 993357, 993676,
   993966, 994774, 995781, 995915, 996117, 996466, 996491, 996762, 997204, 997261,
   997263, 997658, 997850, 998147, 998170, 998414, 998538, 998567, 998679, 998698,
   998992, 999048, 999076, 999153, 999302
]




seeds=[seeds_batch_1,seeds_batch_2,seeds_batch_3,seeds_batch_4,seeds_batch_5,seeds_batch_6,seeds_batch_7,seeds_batch_8,seeds_batch_9,seeds_batch_10,seeds_batch_11,seeds_batch_12,seeds_batch_13,seeds_batch_14,seeds_batch_15,seeds_batch_16]

#seeds=[seeds_batch_1,seeds_batch_2,seeds_batch_3,seeds_batch_4,seeds_batch_5,seeds_batch_6,seeds_batch_7,seeds_batch_8,seeds_batch_9,seeds_batch_10]#,seeds_batch_6,seeds_batch_7,seeds_batch_8,seeds_batch_9,seeds_batch_10]

#seeds=[seeds_batch_1,seeds_batch_2,seeds_batch_3,seeds_batch_4,seeds_batch_5]
#seeds=[seeds_batch_1,seeds_batch_2]


#seeds_batch_1=[123]
#seeds=[seeds_batch_1]

seeds = ak.flatten(seeds)


#seeds=[802239]

print(len(seeds))


#============================= INPUTS =============================

bunches_per_train = 133 #133 #266 #532
#bunches_per_train = 300 #75 #150 #300


# Directory path and filename pattern

# for 250 BL and s.u.
base_path = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS1/ddsim/"
filename_pattern = "ddsim_C3_250_PS1_v2_seed_{}.edm4hep.root"

# for 250 high-L
#base_path = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS2/ddsim/"
#filename_pattern = "ddsim_C3_250_PS2_v2_seed_{}.edm4hep.root"

# for all 550 scenarios
#base_path = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS2/ddsim/"
#filename_pattern = "ddsim_C3_550_PS2_v2_seed_{}_MERGED.edm4hep.root"

# for 250 BL and s.u.
HPP_FILE = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1/ddsim_C3_250_PS1_gg_had_MERGED.edm4hep.root"  # e.g. "/path/to/hpp_background.edm4hep.root"
HPP_MU = 0.059    

# for 250 high-L
#HPP_FILE = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS2/ddsim_C3_250_PS2_gg_had_MERGED.edm4hep.root"  # e.g. "/path/to/hpp_background.edm4hep.root"
#HPP_MU = 0.065    


# for all 550 scenarios
#HPP_FILE = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_550_PS2/ddsim_C3_550_PS2_gg_had_MERGED.edm4hep.root"  # e.g. "/path/to/hpp_background.edm4hep.root"
#HPP_MU = 0.29    


# Plot label parameters
SCENARIO_LABEL = "C³ 250 B.L."  # Shown on left with (N bunches/train)
# B.L. , s.u. , high-L

DETECTOR_VERSION = "SiD_o2_v04"  # Shown on right with subdetector and background type


#============================= END INPUTS =============================







# Validate file paths before opening
print("Validating file paths...")
existing_seeds, missing_seeds = validate_file_paths(base_path, filename_pattern, seeds)

if missing_seeds:
    print(f"Warning: {len(missing_seeds)} files are missing. Proceeding with {len(existing_seeds)} available files.")
    seeds = existing_seeds

# Estimate memory usage for large datasets
if len(seeds) > 50:
    print("\nEstimating memory requirements...")
    memory_stats = estimate_memory_usage(base_path, filename_pattern, seeds, num_samples=5)
    
    # If estimated memory usage is very high, suggest using chunked processing
    if memory_stats['estimated_total_mb'] > memory_stats['available_memory_mb'] * 0.8:
        print(f"Warning: Estimated memory usage ({memory_stats['estimated_total_mb']:.1f} MB) is close to available memory.")
        print("Consider using chunked processing for large datasets.")

# Inspect a sample file to confirm structure without keeping all files open
if len(seeds) > 0:
    sample_seed = seeds[0]
    sample_path = base_path + filename_pattern.format(sample_seed)
    print(f"\nInspecting sample file: {sample_path}")
    try:
        with uproot.open(sample_path) as sample_file:
            sample_tree = sample_file["events"]
            print("Sample tree keys:")
            print(sample_tree.keys())
    except Exception as exc:
        print(f"Warning: Unable to inspect sample file {sample_path}: {exc}")
else:
    print("Error: No seeds available after validation!")
    exit(1)

DETECTOR_CONFIGS = get_detector_configs()
xmls = get_xmls()
main_xml = xmls['main_xml']
vertex_barrel_xml = xmls['vertex_barrel_xml']
vertex_endcap_xml = xmls['vertex_endcap_xml']
tracker_barrel_xml = xmls['tracker_barrel_xml']
tracker_endcap_xml = xmls['tracker_endcap_xml']
tracker_forward_xml = xmls['tracker_forward_xml']
ecal_barrel_xml = xmls['ecal_barrel_xml']
ecal_endcap_xml = xmls['ecal_endcap_xml']
hcal_barrel_xml = xmls['hcal_barrel_xml']
hcal_endcap_xml = xmls['hcal_endcap_xml']
beamcal_xml = xmls['beamcal_xml']
lumical_xml = xmls['lumical_xml']
muon_barrel_xml = xmls['muon_barrel_xml']
muon_endcap_xml = xmls['muon_endcap_xml']

# ============================= CELL SIZE OVERRIDES (mm) =============================
# Override per-subdetector cell sizes using the table-provided values
# Barrel
DETECTOR_CONFIGS['SiVertexBarrel'].cell_sizes['default'] = {'x': 0.010, 'y': 0.010}
#DETECTOR_CONFIGS['SiVertexBarrel'].cell_sizes['default'] = {'x': 0.034, 'y': 0.034}
DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes['default'] = {'x': 0.025,  'y': 0.100}
DETECTOR_CONFIGS['ECalBarrel'].cell_sizes['default']    = {'x': 0.025,  'y': 0.100}
DETECTOR_CONFIGS['HCalBarrel'].cell_sizes['default']    = {'x': 30.0,  'y': 30.0}
DETECTOR_CONFIGS['MuonBarrel'].cell_sizes['default']    = {'x': 41.0,  'y': 5500.0}



# Endcap / Forward
DETECTOR_CONFIGS['SiVertexEndcap'].cell_sizes['default']   = {'x': 0.010, 'y': 0.010}
DETECTOR_CONFIGS['SiTrackerForward'].cell_sizes['default'] = {'x': 0.010, 'y': 0.010}
#DETECTOR_CONFIGS['SiVertexEndcap'].cell_sizes['default']   = {'x': 0.030, 'y': 0.030}
#DETECTOR_CONFIGS['SiTrackerForward'].cell_sizes['default'] = {'x': 0.030, 'y': 0.030}


DETECTOR_CONFIGS['SiTrackerEndcap'].cell_sizes['default']  = {'x': 0.025,  'y': 0.100}
DETECTOR_CONFIGS['ECalEndcap'].cell_sizes['default']       = {'x': 0.025,  'y': 0.100}
DETECTOR_CONFIGS['HCalEndcap'].cell_sizes['default']       = {'x': 30.0, 'y': 30.0}
DETECTOR_CONFIGS['MuonEndcap'].cell_sizes['default']       = {'x': 41.0, 'y': 5500.0}
DETECTOR_CONFIGS['LumiCal'].cell_sizes['default']          = {'x': 2.5,  'y': 2.5}
DETECTOR_CONFIGS['BeamCal'].cell_sizes['default']          = {'x': 5.0,  'y': 5.0}


# ===================================================================================



# # ============================= CELL SIZE OVERRIDES (mm) =============================
# # Override per-subdetector cell sizes using the table-provided values
# # Barrel
# DETECTOR_CONFIGS['SiVertexBarrel'].cell_sizes['default'] = {'x': 10, 'y': 10}
# DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes['default'] = {'x': 3,  'y': 3}
# DETECTOR_CONFIGS['ECalBarrel'].cell_sizes['default']    = {'x': 5,  'y': 5}
# DETECTOR_CONFIGS['HCalBarrel'].cell_sizes['default']    = {'x': 30,  'y': 30}
# DETECTOR_CONFIGS['MuonBarrel'].cell_sizes['default']    = {'x': 10,  'y': 41}

# # Endcap / Forward
# DETECTOR_CONFIGS['SiVertexEndcap'].cell_sizes['default']   = {'x': 10, 'y': 10}
# DETECTOR_CONFIGS['SiTrackerEndcap'].cell_sizes['default']  = {'x': 0.25, 'y': 0.25}
# DETECTOR_CONFIGS['SiTrackerForward'].cell_sizes['default'] = {'x': 0.10, 'y': 0.10}
# DETECTOR_CONFIGS['ECalEndcap'].cell_sizes['default']       = {'x': 0.50, 'y': 0.50}
# DETECTOR_CONFIGS['HCalEndcap'].cell_sizes['default']       = {'x': 30.0, 'y': 30.0}
# DETECTOR_CONFIGS['MuonEndcap'].cell_sizes['default']       = {'x': 10.0, 'y': 41.0}
# DETECTOR_CONFIGS['LumiCal'].cell_sizes['default']          = {'x': 2.5,  'y': 2.5}
# DETECTOR_CONFIGS['BeamCal'].cell_sizes['default']          = {'x': 5.0,  'y': 5.0}


# # ===================================================================================


# Try each detector in turn
# Try analyzing all detectors again
detectors_to_analyze = [
    ('SiVertexBarrel', vertex_barrel_xml),
    #('SiVertexEndcap', vertex_endcap_xml),
    # ('SiTrackerBarrel', tracker_barrel_xml),
    # ('SiTrackerEndcap', tracker_endcap_xml),
    #('SiTrackerForward', tracker_forward_xml),
    # ('ECalBarrel', ecal_barrel_xml),
    # ('ECalEndcap', ecal_endcap_xml),
    # ('HCalBarrel', hcal_barrel_xml),
    # ('HCalEndcap', hcal_endcap_xml),
    # ('BeamCal', beamcal_xml),
    # ('LumiCal', lumical_xml),
    # ('MuonBarrel', muon_barrel_xml),
    # ('MuonEndcap', muon_endcap_xml)
]


# Print constants for reference
print("Constants from main XML:")
constants = parse_detector_constants(main_xml)
for name, value in sorted(constants.items()):
    if any(det in name for det in ['Vertex', 'Tracker', 'ECal', 'HCal', 'Muon', 'BeamCal', 'LumiCal']):
        print(f"{name}: {value}")


print("\nAnalyzing detectors:")
for detector_name, xml_file in detectors_to_analyze:
    print(f"\nAnalyzing {detector_name}...")
    try:

        detector_config = DETECTOR_CONFIGS[detector_name]
        # Pass detector name to get specific debug info
        constants = parse_detector_constants(main_xml, detector_name)
        geometry_info = get_geometry_info(xml_file, detector_config, constants=constants)
        
        print("Jim: geometry info = ", geometry_info)

        print(f"Total cells: {geometry_info['total_cells']}")
        print("Layers:")
        for layer, info in sorted(geometry_info['layers'].items()):
            print(f"  Layer {layer}:")
            for key, value in sorted(info.items()):
                if key not in ['cells_per_module', 'module_type']:
                    print(f"    {key}: {value}")
    except Exception as e:
        print(f"Error processing {detector_name}: {str(e)}")
        traceback.print_exc()



time_cut = -1 # 5.25*133 #-1 #100 # in ns
calo_hit_time_def = 1

#Jim: default energy value in edm4hep is in GeV!!!
custom_thresholds = {
    'silicon': 0.6e-6,               # 0.6 keV for silicon (global default)
    'ecal_hits': 50e-6,              # 50 keV for ECAL hits (global default)
    'ecal_contributions': 1e-20,     # very small for ECAL contributions (global default)
    'hcal_hits': 0.24e-3,            # 0.24 MeV for HCAL hits (global default)
    'hcal_contributions': 1e-20,     # very small for HCAL contributions (global default)
    #Per-detector overrides (examples): use exact detector names
    #Silicon examples
    'SiVertexBarrel_silicon': 0.6e-6, # 0.6 keV
    'SiVertexEndcap_silicon': 0.5e-6, # 0.5 keV
    'SiTrackerBarrel_silicon': 3e-5, # 30 keV
    'SiTrackerEndcap_silicon': 3e-5, # 30 keV
    'SiTrackerForward_silicon': 0.4e-6, # 0.4 keV
    #ECAL examples (hits and contribution thresholds)
    'ECalBarrel_hits': 5e-5, # 50 keV
    'ECalBarrel_contributions': 1e-20,
    'ECalEndcap_hits': 5e-5, # 50 keV
    'ECalEndcap_contributions': 1e-20,
   # HCAL examples
    'HCalBarrel_hits': 0.24e-3, # 0.24 MeV
    'HCalBarrel_contributions': 1e-20,
    'HCalEndcap_hits': 0.23e-3, # 0.23 MeV
    'HCalEndcap_contributions': 1e-20,
    #Forward calorimeters
    'BeamCal_hits': 5e-5, # 50 keV
    'BeamCal_contributions': 1e-20,
    'LumiCal_hits': 4e-5, # 40 keV
    'LumiCal_contributions': 1e-20,
}



# #Jim: default energy value in edm4hep is in GeV!!!
# custom_thresholds = {
#     'silicon': 0.0,               # 0.6 keV for silicon (global default)
#     'ecal_hits': 0.0,              # 50 keV for ECAL hits (global default)
#     'ecal_contributions': 0.0,     # very small for ECAL contributions (global default)
#     'hcal_hits': 0.0,            # 0.24 MeV for HCAL hits (global default)
#     'hcal_contributions': 0.0,     # very small for HCAL contributions (global default)
#     #Per-detector overrides (examples): use exact detector names
#     #Silicon examples
#     'SiVertexBarrel_silicon': 0.0, # 0.6 keV
#     'SiVertexEndcap_silicon': 0.0, # 0.5 keV
#     'SiTrackerBarrel_silicon': 0.0, # 30 keV
#     'SiTrackerEndcap_silicon': 0.0, # 30 keV
#     'SiTrackerForward_silicon': 0.0, # 0.4 keV
#     #ECAL examples (hits and contribution thresholds)
#     'ECalBarrel_hits': 0.0, # 50 keV
#     'ECalBarrel_contributions': 0.0,
#     'ECalEndcap_hits': 0.0, # 50 keV
#     'ECalEndcap_contributions': 0.0,
#    # HCAL examples
#     'HCalBarrel_hits': 0.0, # 0.24 MeV
#     'HCalBarrel_contributions': 0.0,
#     'HCalEndcap_hits': 0.0, # 0.23 MeV
#     'HCalEndcap_contributions': 0.0,
#     #Forward calorimeters
#     'BeamCal_hits': 0.0, # 50 keV
#     'BeamCal_contributions': 0.0,
#     'LumiCal_hits': 0.0, # 40 keV
#     'LumiCal_contributions': 0.0,
# }










# Enable vectorized processing for performance
USE_VECTORIZED_PROCESSING = True

# Per-detector effective cluster size multipliers applied at plotting stage
CLUSTER_SIZE = {
    'SiVertexBarrel': 3,
    'SiVertexEndcap': 3,
    'SiTrackerBarrel': 1,
    'SiTrackerEndcap': 1,
    'SiTrackerForward': 3,
    'ECalBarrel': 1,
    'ECalEndcap': 1,
    'HCalBarrel': 1,
    'HCalEndcap': 1,
    'BeamCal': 1,
    'LumiCal': 1,
    'MuonBarrel': 1,
    'MuonEndcap': 1,
}

# Safety factor applied per detector (uniform value of 2 for all detectors)
SAFETY_FACTOR = {det: 2 for det in CLUSTER_SIZE.keys()}

# Optional per-detector y-axis limits for occupancy vs buffer depth plots
# Example: {'SiVertexBarrel': (1e-6, 1e-1), 'SiVertexEndcap': (1e-6, 5e-2)}
OCCUPANCY_YLIMS = {
    'SiVertexBarrel': (1e-9, 1e-2),
    'SiVertexEndcap': (1e-6, 1e-3),
    'SiTrackerBarrel': (1e-8, 1e-4),
    'SiTrackerEndcap': (5e-8, 5e-5),
    'SiTrackerForward': (1e-7, 5e-3),
}
    
# Limit concurrent train processing to avoid exhausting file descriptors
TRAIN_BATCH_SIZE = 3 # how many batches to split the total numer of trains into
MAX_TRAIN_WORKERS = 30 # how many workers to use to process each train batch



print(f"Performance optimizations enabled:")
print(f"  - Vectorized hit processing: {'✓' if USE_VECTORIZED_PROCESSING else '✗'}")


detectors_to_analyze = [
   ('SiVertexBarrel', vertex_barrel_xml),
   ('SiVertexEndcap', vertex_endcap_xml),
   #   ('SiTrackerBarrel', tracker_barrel_xml),
   #   ('SiTrackerEndcap', tracker_endcap_xml),
   ('SiTrackerForward', tracker_forward_xml),
   #   ('ECalBarrel', ecal_barrel_xml),
   #   ('ECalEndcap', ecal_endcap_xml),
   #   ('HCalBarrel', hcal_barrel_xml),
   #   #('HCalEndcap', hcal_endcap_xml),
   #   ('BeamCal', beamcal_xml),
   #   ('LumiCal', lumical_xml),
   #   ('MuonBarrel', muon_barrel_xml),
   #   ('MuonEndcap', muon_endcap_xml)
 ]

OCCUPANCY_SCALING_FACTORS = {
    det_name: SAFETY_FACTOR.get(det_name, 1.0) * CLUSTER_SIZE.get(det_name, 1.0)
    for det_name, _ in detectors_to_analyze
}







 # detectors_to_analyze = [
 #     ('SiVertexBarrel', vertex_barrel_xml)
 # ]
   
 # analyze_detectors_and_plot(DETECTOR_CONFIGS=DETECTOR_CONFIGS,
 #                            detectors_to_analyze=detectors_to_analyze,
 #                            event_trees=events_trees,
 #                            main_xml=main_xml,
 #                            remove_zeros=1,
 #                            time_cut=time_cut,
 #                            calo_hit_time_def=calo_hit_time_def, # Use cumulative energy for time
 #                            energy_thresholds=custom_thresholds)
 
events_trees_by_train = analyze_detectors_and_plot_by_train(
     DETECTOR_CONFIGS=DETECTOR_CONFIGS,
     detectors_to_analyze=detectors_to_analyze,
     all_seeds=seeds,
     bunches_per_train=bunches_per_train,
     main_xml=main_xml,
     base_path=base_path,
     filename_pattern=filename_pattern,
     remove_zeros=True,
     time_cut=time_cut,
     calo_hit_time_def=calo_hit_time_def,
     energy_thresholds=custom_thresholds,
     hpp_file=HPP_FILE,
     hpp_mu=HPP_MU,
     nlayer_batch=1,
     scenario_label=SCENARIO_LABEL,
     detector_version=DETECTOR_VERSION,
     use_vectorized_processing=USE_VECTORIZED_PROCESSING,
     occupancy_ylim_map=OCCUPANCY_YLIMS,
     occupancy_scaling_map=OCCUPANCY_SCALING_FACTORS,
     train_batch_size=TRAIN_BATCH_SIZE,
     max_train_workers=MAX_TRAIN_WORKERS
 )

 
 # print(f"Completed train-based analysis with {len(events_trees_by_train)} trains of {bunches_per_train} bunches each")
# detectors_to_analyze = [
#      ('SiTrackerBarrel', tracker_barrel_xml),
#      ('SiTrackerEndcap', tracker_endcap_xml),
#      ('SiTrackerForward', tracker_forward_xml)
#  ]
# events_trees_by_train = analyze_detectors_and_plot_by_train(
#      DETECTOR_CONFIGS=DETECTOR_CONFIGS,
#      detectors_to_analyze=detectors_to_analyze,
#      all_seeds=seeds,
#      bunches_per_train=bunches_per_train,
#      main_xml=main_xml,
#      base_path=base_path,
#      filename_pattern=filename_pattern,
#      remove_zeros=True,
#      time_cut=time_cut,
#      calo_hit_time_def=calo_hit_time_def,
#      energy_thresholds=custom_thresholds,
#      hpp_file=HPP_FILE,
#      hpp_mu=HPP_MU,
#      nlayer_batch=1,
#      scenario_label=SCENARIO_LABEL,
#      detector_version=DETECTOR_VERSION
#  )



# detectors_to_analyze = [
#      ('ECalBarrel', ecal_barrel_xml),
#      ('ECalEndcap', ecal_endcap_xml),
#      ('HCalBarrel', hcal_barrel_xml),
#      #('HCalEndcap', hcal_endcap_xml)
#  ]
# events_trees_by_train = analyze_detectors_and_plot_by_train(
#      DETECTOR_CONFIGS=DETECTOR_CONFIGS,
#      detectors_to_analyze=detectors_to_analyze,
#      all_seeds=seeds,
#      bunches_per_train=bunches_per_train,
#      main_xml=main_xml,
#      base_path=base_path,
#      filename_pattern=filename_pattern,
#      remove_zeros=True,
#      time_cut=time_cut,
#      calo_hit_time_def=calo_hit_time_def,
#      energy_thresholds=custom_thresholds,
#      hpp_file=HPP_FILE,
#      hpp_mu=HPP_MU,
#      nlayer_batch=10,
#      scenario_label=SCENARIO_LABEL,
#      detector_version=DETECTOR_VERSION
#  )
# detectors_to_analyze = [
#      ('MuonBarrel', muon_barrel_xml),
#      ('MuonEndcap', muon_endcap_xml)
#  ]
# events_trees_by_train = analyze_detectors_and_plot_by_train(
#      DETECTOR_CONFIGS=DETECTOR_CONFIGS,
#      detectors_to_analyze=detectors_to_analyze,
#      all_seeds=seeds,
#      bunches_per_train=bunches_per_train,
#      main_xml=main_xml,
#      base_path=base_path,
#      filename_pattern=filename_pattern,
#      remove_zeros=True,
#      time_cut=time_cut,
#      calo_hit_time_def=calo_hit_time_def,
#      energy_thresholds=custom_thresholds,
#      hpp_file=HPP_FILE,
#      hpp_mu=HPP_MU,
#      nlayer_batch=10,
#      scenario_label=SCENARIO_LABEL,
#      detector_version=DETECTOR_VERSION
#  )

# detectors_to_analyze = [
#     ('BeamCal', beamcal_xml),
#     ('LumiCal', lumical_xml)
# ]

# events_trees_by_train = analyze_detectors_and_plot_by_train(
#     DETECTOR_CONFIGS=DETECTOR_CONFIGS,
#     detectors_to_analyze=detectors_to_analyze,
#     all_seeds=seeds,
#     bunches_per_train=bunches_per_train,
#     main_xml=main_xml,
#     base_path=base_path,
#     filename_pattern=filename_pattern,
#     remove_zeros=True,
#     time_cut=time_cut,
#     calo_hit_time_def=calo_hit_time_def,
#     energy_thresholds=custom_thresholds,
#     hpp_file=HPP_FILE,
#     hpp_mu=HPP_MU,
#     nlayer_batch=10,
#     scenario_label=SCENARIO_LABEL,
#     detector_version=DETECTOR_VERSION
# )
