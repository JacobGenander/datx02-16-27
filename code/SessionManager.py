import argparse
import ConfigParser
import os
import getpass  # Reading current username
import pprint
import sys
from LSTM import run

class Parameters:

    def __init__(self):
        env_file = "environment.ini"
        paths_section = "data_paths"
        env = ConfigParser.ConfigParser()
        env.readfp(open(env_file))
        user_paths = env.items(paths_section)
        username = getpass.getuser()
        self.data_path = None
        for (user, path) in user_paths:
            if username == user:
                self.data_path = path
                break

        if self.data_path == None:
            print("Error! no data_path specified for user \"{}\"".format(username))
            print("In the file {}, please specify where data is located with \
                  \n[{}]\n{} = /data/folder".format(env_file, paths_section, username))
            sys.exit(1)
        else:
            print("data_path:\t{}".format(self.data_path))
#        self.training_file
#        self.validation_file

    def read_parameters(self, filename):
        self.parameters = dict()
        self.parameters["session_store"] = os.path.dirname(os.path.abspath(filename))
        self.parameters["session_tf_logs"] = os.path.join(
            self.parameters["session_store"], "tf_logs")
        self.parameters["session_csv_logs"] = os.path.join(
            self.parameters["session_store"], "log.csv")
        self.parameters["session_model"] = os.path.join(
            self.parameters["session_store"], "model.ckpt")

        print("Models and logfiles will be saved at {}".format(self.parameters["session_store"]))
        params = ConfigParser.ConfigParser()
        params.readfp(open(filename))
        files = params.items("files")
        for (name, path) in files:
            #print("Joining {} and {}".format(self.data_path, path))
            self.parameters[name] = os.path.join(self.data_path, path)
            if not os.path.isfile(self.parameters[name]):
                print("Warning! The file for \'{}\': \"{}\" does not exist"
                      .format(name, self.parameters[name]))

        hyperparameters = params.items("hyperparameters")
        self.parameters.update(dict(hyperparameters))
        print("----- The following parameters have been set -----")
        pprint.pprint(self.parameters)
        return self.parameters

def main():
    parser = argparse.ArgumentParser("Run session")

    parser.add_argument("filename",
                        help="Start session by supplying a \"parameters.ini\" file",
                        metavar="/path/to/parameters.ini")
    args = parser.parse_args()
    params = Parameters()
    params_dict = params.read_parameters(args.filename)

    raw_input("Press enter to start script with the configuration above\n")

    run(params_dict)
main()
