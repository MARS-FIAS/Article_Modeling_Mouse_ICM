# Article_Modeling_Mouse_ICM

## AI-powered simulation-based inference of a genuinely spatial-stochastic gene regulation model of early mouse embryogenesis

**_Preprint_ Article Link** [_ARXIV_]

[https://doi.org/10.48550/arXiv.2402.15330](https://doi.org/10.48550/arXiv.2402.15330)

__Partial Data Bank Link__ [_Zenodo_]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12637055.svg)](https://doi.org/10.5281/zenodo.12637055)

__Complete Data Bank Links__ [_Zenodo_]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13891602.svg)](https://doi.org/10.5281/zenodo.13891602)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13896947.svg)](https://doi.org/10.5281/zenodo.13896947)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13896989.svg)](https://doi.org/10.5281/zenodo.13896989)

> [!NOTE]
> Download the whole repository, and make sure to preserve the directory structure. The corresponding Python scripts automatically find the supporting files based on their relative repository location. The *requirements.txt* file lists the minimum prerequisites for executing these scripts. The *repository_tree.html* file provides a quick view of the directory structure.

All the code files are under the folder *Script_Bank*. The subfolder *Analysis* contains figure-generating Python scripts, and files required to produce data sets for all corresponding computational/simulation experiments. The user should mainly play with these Python files. If the user wishes to regenerate the training data for the ANN (SBI workflow), then the subfolders *Prime* and *HPC* contain the necessary files; contact us for complete instructions.

The data sets must be uncompressed, extracted, and placed under the corresponding subfolders of *Data_Bank*. Please use the data set placement guidelines as described over the following sections.

> [!WARNING]
> The simulation-inference data bank is not available from this GitHub repository. Use the links provided at the beginning of this *README* file. As of now, the complete data bank is available at Zenodo: [https://zenodo.org/](https://zenodo.org/). Please note, the complete (uncompressed) data bank is around 500 GBs.

----

## AVAILABLE FIGURE DATA SETS

> PATH = ./Article_Modeling_Mouse_ICM

### Figure 2

 - This figure was partially made using LibreOffice Draw and Inkscape!

 - Scripts

   - PATH/Script_Bank/Analysis/Fig2[E].py

   - PATH/Script_Bank/Analysis/Fig2[F].py

   - PATH/Script_Bank/Analysis/Fig2_Services.py

 - Data Sets [Placement]

   - Fig2_ITWT.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

   - Fig2_RTM.zip

       PATH/Data_Bank/Shallow_Grid_1_Rule_Kern/Observe_1/

### Figure 3

 - Script

   - PATH/Script_Bank/Analysis/Fig3.py

 - Data Set [Placement]

   - Fig3.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 4

 - Scripts

   - PATH/Script_Bank/Analysis/Fig4.py

   - PATH/Script_Bank/Analysis/Fig4_Auxiliary.py

 - Data Set [Placement]

   - Fig4.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 5

 - Script

   - PATH/Script_Bank/Analysis/Fig5.py

 - Data Sets [Placement]

   - Fig5_ITWT.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

   - Fig5_RTM.zip

       PATH/Data_Bank/Shallow_Grid_1_Rule_Kern/Observe_1/

### Figure 6

 - Script

   - PATH/Script_Bank/Analysis/Fig6.py

 - Data Set [Placement]

   - Fig6.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 7

 - Scripts

   - PATH/Script_Bank/Analysis/Fig7_Alp.py

   - PATH/Script_Bank/Analysis/Fig7_Bet.py

 - Data Sets [Placement]

   - Fig7_Alp.zip

   - Fig7_Bet.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 8

 - Script

   - PATH/Script_Bank/Analysis/Fig8.py

 - Data Set [Placement]

   - Fig8.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 9

 - Scripts

   - PATH/Script_Bank/Analysis/Fig9_Alp.py

   - PATH/Script_Bank/Analysis/Fig9_Bet.py

 - Data Sets [Placement]

   - {Fig9_Alp_Card_1.zip, Fig9_Alp_Card_5.zip, Fig9_Alp_Card_10.zip, Fig9_Alp_Card_25.zip, Fig9_Alp_Card_50.zip, Fig9_Alp_Card_75.zip, Fig9_Alp_Card_100.zip}

   - {Fig9_Bet_Card_1.zip, Fig9_Bet_Card_5.zip, Fig9_Bet_Card_10.zip, Fig9_Bet_Card_25.zip, Fig9_Bet_Card_50.zip, Fig9_Bet_Card_75.zip, Fig9_Bet_Card_100.zip}

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 10

 - Script

   - PATH/Script_Bank/Analysis/Fig10.py

 - Data Sets [Placement]

   - {Fig10_Alp_Wait_0.zip, Fig10_Alp_Wait_4.zip, Fig10_Alp_Wait_8.zip, Fig10_Alp_Wait_12.zip, Fig10_Alp_Wait_16.zip, Fig10_Alp_Wait_24.zip, Fig10_Alp_Wait_32.zip, Fig10_Alp_Wait_40.zip}

   - {Fig10_Bet_Wait_0.zip, Fig10_Bet_Wait_4.zip, Fig10_Bet_Wait_8.zip, Fig10_Bet_Wait_12.zip, Fig10_Bet_Wait_16.zip, Fig10_Bet_Wait_24.zip, Fig10_Bet_Wait_32.zip, Fig10_Bet_Wait_40.zip}

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 11

 - Script

   - PATH/Script_Bank/Analysis/Fig11.py

 - Data Sets [Placement]

   - {Fig11_Alp_Wait_0.zip, Fig11_Alp_Wait_4.zip, Fig11_Alp_Wait_8.zip, Fig11_Alp_Wait_12.zip, Fig11_Alp_Wait_16.zip, Fig11_Alp_Wait_24.zip, Fig11_Alp_Wait_32.zip, Fig11_Alp_Wait_40.zip}

   - {Fig11_Bet_Wait_0.zip, Fig11_Bet_Wait_4.zip, Fig11_Bet_Wait_8.zip, Fig11_Bet_Wait_12.zip, Fig11_Bet_Wait_16.zip, Fig11_Bet_Wait_24.zip, Fig11_Bet_Wait_32.zip, Fig11_Bet_Wait_40.zip}

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

### Figure 14 AND Figure 15

 - Scripts

   - PATH/Script_Bank/Analysis/Fig14_Fig15.py

   - PATH/Script_Bank/Analysis/Fig14_Fig15_Auxiliary.py

 - Data Sets [Placement]

   - Fig14_Fig15_ITWT.zip

   - Fig14_Fig15_Auxiliary.zip

       PATH/Data_Bank/Shallow_Grid_1_N_Link/Observe_1/

   - Fig14_Fig15_RTM.zip

       PATH/Data_Bank/Shallow_Grid_1_Rule_Kern/Observe_1/

----

## INACCESSIBLE FIGURE DATA SETS (ONLY AVAILABLE BY REQUEST)

### Figure 1

 - This figure was partially made using LibreOffice Draw and Inkscape!

### Figure 12

 - This figure was made using LibreOffice Draw and Inkscape!

### Figure 13

 - This figure was made using LibreOffice Draw and Inkscape!
