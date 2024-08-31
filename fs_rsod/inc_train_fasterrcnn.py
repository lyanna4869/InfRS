
if __name__ == "__main__":
    import sys
    from fs_rsod.train_fasterrcnn import main

    args = sys.argv[1:]
    if len(args) == 0:
        ## test
        # args = [
        # "--config-file", "configs/RSOD/base_training/split1.yml", "--start-iter", "0",
        # "--eval-only", "--eval-iter", "model_best"
        # ]
        ## novel
        args = [
            "--config-file", "configs/RSOD/split3/inc/3shot_dmk.yml", "--start-iter", "0"
        ]
    main(args)
