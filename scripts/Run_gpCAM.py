################################################################
######Run gpCAM#################################################
################################################################
import sys
import traceback
def main():
    import gpcam
    import Config
    gpcam.global_config = Config
    from gpcam import global_config
    from gpcam.main import main as main2
    from pathlib import Path
    #license here
    input()

    if len(sys.argv) == 1:
        main2()
    if len(sys.argv) > 1:
        if "-d" in sys.argv and "-h" in sys.argv:
            data_index = sys.argv.index("-d")
            hp_index = sys.argv.index("-h")
            if hp_index < data_index: exit("give data (-d) first")
            data_files = sys.argv[data_index+1:hp_index]
            hp_files = sys.argv[hp_index+1:]
            print("data files given in command line:", data_files)
            print("hyper parameter files given in command line:", hp_files)
            main2(init_data_files = data_files, init_hyperparameter_files = hp_files)
        elif "-d" in sys.argv:
            data_index = sys.argv.index("-d")
            data_files = sys.argv[data_index+1:]
            print("data files given in command line:", data_files)
            main2(init_data_files = data_files, init_hyperparameter_files = None)
        elif "-h" in sys.argv:
            hp_index = sys.argv.index("-h")
            hp_files = sys.argv[hp_index+1:]
            print("hyper parameter files given in command line:", hp_files)
            main2(init_data_files = None, init_hyperparameter_files = hp_files)
        else:
            exit("no valid file option given")
if __name__ == "__main__":
    try:
        main()
        logf = open("errors.log", "w")
    except:
        print("gpCAM FAILED")
        print("see ./scripts/errors.log for details")
        print("======================")
        logf = open("errors.log", "w")
        traceback.print_exc(file = logf)
        #traceback.print_exc()
