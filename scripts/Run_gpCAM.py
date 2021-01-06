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
    print("\n\
    *** License Agreement ***\n\
    \n\
    GPL v3 License\n\
    \n\
    gpCAM Copyright (c) 2021, The Regents of the University of California,\n\
    through Lawrence Berkeley National Laboratory (subject to receipt of\n\
    any required approvals from the U.S. Dept. of Energy). All rights reserved.\n\
    \n\
    This program is free software: you can redistribute it and/or modify\n\
    it under the terms of the GNU General Public License as published by\n\
    the Free Software Foundation, either version 3 of the License, or\n\
    (at your option) any later version.\n\
    \n\
    This program is distributed in the hope that it will be useful,\n\
    but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
    GNU General Public License for more details.\n\
    \n\
    You should have received a copy of the GNU General Public License\n\
    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\
    \n\
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'\n\
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\n\
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\n\
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE\n\
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\n\
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR \n\
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\n\
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR\n\
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF\n\
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\
    ")
    input("Press ENTER to agree to the terms of the license and to continue.")


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
            print("hyperparameter files given in command line:", hp_files)
            main2(init_data_files = data_files, init_hyperparameter_files = hp_files)
        elif "-d" in sys.argv:
            data_index = sys.argv.index("-d")
            data_files = sys.argv[data_index+1:]
            print("data files given in command line:", data_files)
            main2(init_data_files = data_files, init_hyperparameter_files = None)
        elif "-h" in sys.argv:
            hp_index = sys.argv.index("-h")
            hp_files = sys.argv[hp_index+1:]
            print("hyperparameter files given in command line:", hp_files)
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
