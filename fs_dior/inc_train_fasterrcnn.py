from fs_dior.train_fasterrcnn import main

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) == 0:
        ## test
        # args = [
        # "--config-file", "configs/DIOR/base_training/split1.yml", "--start-iter", "0",
        # "--eval-only", "--eval-iter", "model_best"
        # ]
        ## novel
        args = [
            "--config-file", "configs/DIOR/split3/inc/3shot_dmk.yml", "--start-iter", "0"
        ]
    main(args)
