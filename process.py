import musicbox.processor
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "input image file, must be able to open with cv2.imread")
    parser.add_argument("-d", "--debug-dir", dest = "debug", default = "", help = "Where to write debug files to")
    parser.add_argument("-v", "--verbose", action = "store_true", help = "Enable verbose output")
    parser.add_argument("-c", "--config", help = "config file containing required information about plate type",
                        const = "config.yaml", default = "config.yaml", nargs = "?")
    parser.add_argument("-t", "--type", help = "type of the plate to process",
                        const = "ariston", default = "ariston", nargs = "?")                                                       
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("{:>10}".format("FAIL"))
            print("Could not read config file, original error:", exc)

    config = config[args.type]

    our_processor = musicbox.processor.Processor.from_file(args.input, config, debug_dir = args.debug, verbose = True)

    our_processor.run()

if __name__ == "__main__":
    main()