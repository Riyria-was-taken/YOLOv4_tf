def detect():
    print("detecting...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True
    subparsers.add_parser("detect")
    subparsers.add_parser("train")
    subparsers.add_parser("verify")

    args = parser.parse_args()

    if args.action == "detect":
        detect()
    else:
        print("The " + args.action + " action is not yet implemented :<")
