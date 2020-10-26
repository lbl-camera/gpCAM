################################################################
######Run Visualizations########################################
################################################################
import sys
import gpcam
import Config
gpcam.global_config = Config
from gpcam import global_config
from gpcam.visualize import main
from pathlib import Path
#license here
if len(sys.argv) == 1:
    print("please specify data file via command line parameter, e.g. python Run_Visualization ../data/current_data/Data.npy");exit()
if len(sys.argv) == 2:
    main(data_path = sys.argv[1])
if len(sys.argv) == 3:
    main(data_path = sys.argv[1], hyperparameter_path = sys.argv[2])
