from fs_nwpu.train_fasterrcnn import main

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) == 0:
        ## novel
        args = [
            "--config-file", "configs/NWPU/split3/dmk/3shot_dmk.yml", "--start-iter", "0"
        ]
    main(args)
