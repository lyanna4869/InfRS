import torch
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys
from fs.faster_rcnn import Trainer
Tester = Trainer
from fsdet.engine import default_argument_parser, default_setup as setup
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.utils.events import EventStorage
from fsdet.evaluation import DatasetEvaluator, inference_on_dataset, print_csv_format, inference_context
from fs.core.data import build_detection_train_loader
from collections import OrderedDict
from fsdet.config import globalvar as gv
import logging, time, datetime
logger = logging.getLogger()


def flops_on_dataset(model, data_loader, evaluator, logging_interval = 50):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images(batches)".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            # outputs = model(inputs)
            image_data = inputs[0]['image'].unsqueeze_(0)
            image_data = image_data.to(model.device)
            flops = FlopCountAnalysis(model, (image_data, ))
            print("FLOPs: ", flops.total())

                # 分析parameters
            print(parameter_count_table(model))
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            # evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )
            break

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

class NwpuFlopCalcer(Tester):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = None
        
        return build_detection_train_loader(cfg, mapper=mapper)

    def calc_flop(self, cfg, model, evaluators=None, test_flop=True):
        model.eval()
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = NwpuFlopCalcer.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = NwpuFlopCalcer.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if test_flop:
                results_i = flops_on_dataset(model, data_loader, evaluator, cfg.LOGGING_INTERVAL)
            else:
            ### 测试 FPS
                results_i = inference_on_dataset(model, data_loader, evaluator, cfg.LOGGING_INTERVAL)
            results[dataset_name] = results_i
           
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def train(self, start_iter, max_iter):
        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                data = next(self._data_loader_iter)
                data[0].pop('instances')
                data[0].pop('height')
                data[0].pop('width')
                print(data[0]['image'].shape)
                flops = FlopCountAnalysis(self.model, (data, ))
                print("FLOPs: ", flops.total())

                # 分析parameters
                print(parameter_count_table(self.model))
                # flops, params, layer_info = profile(self.model, (data, ), ret_layer_info=True)
                # # self.run_step()
                # flops, params = clever_format([flops, params], "%.3f")
                # print('flops: ', flops, 'params: ', params, )
                # print("=====================")
                # for lk, lv in layer_info.items():
                #     print(lk, lv)
                # print("=====================")
                self.after_step()
            self.after_train()
def main(args=None):
    # net = Model()  # 定义好的网络模型
    # inputs = torch.randn(1, 3, 1024, 1024)
    # flops, params = profile(net, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    cfg = setup(args)
    import nwpu.core as core
    core.COMMON_CONFG["CROP_EN"] = cfg.INPUT.CROP.ENABLED
    core.builtin.register_all()
    if args.eval_name is not None:
        ckpt_file = os.path.join(cfg.OUTPUT_DIR, f'{args.eval_name}.pth')
    else:
        # load checkpoint at last iteration
        ckpt_file = cfg.MODEL.WEIGHTS
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    model = Tester.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            ckpt_file, resume=False)
    calcer = NwpuFlopCalcer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()
    calcer.calc_flop(cfg, model, test_flop=args.test_flop)
    

if __name__ == "__main__":
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    parser = default_argument_parser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--eval-name", type=str, default=None)
    parser.add_argument("--test-flop", action="store_true")
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "--eval-only",
            "--config-file", "configs/NWPU/split1/flop_wf101_10shot_CL_IoU.yml",
            # "--eval-name", "model_best"
        ]
    args = parser.parse_args(args)
    print("Command Line Args:", args)
    if args.test_flop:
        gv.for_calculate_flops = True
        gv.rcnn_inference_pre_process = False
        gv.rcnn_inference_post_process = False
    else:
        gv.for_calculate_flops = False

    main(args)
