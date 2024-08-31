## 新增加的全局配置，方便修改 tsne 或者计算 FLOPs 

# instances_per_annotations： 
# 默认值 0 无限制, 大于 0 则每个图片中的实例数不超过该值
# 调 tsne 的时候设置该值为 1
instances_per_annotations = 0

## 默认为 None，普通模式，"tsne" 为 tsne 模式下得收集 feature, "rpn" 为 收集 rpn
collect_roi_feature_stage = None

# tsne_manaual_select_instances 手动选择样本，方便绘图或者曲线
# 默认值 False, 当需要手动选择样本或者调试 tsne 时设为 True
tsne_manaual_select_instances = False
# tsne_select_instance_count: 在加载数据时，每个类别最多选取 x 个样本进行 tsne 绘制
tsne_select_instance_count = 20
# tsne_desired_annos 手动指定 instance 的 id 号，每个元素为 int 类型
tsne_desired_annos = []

# rcnn 在推理时进行预处理，当绘制图片，或计算 时可根据情况关闭
rcnn_inference_pre_process = True
# rcnn 在推理时进行后处理，当绘制图片，或保存 image feature embedding 时可根据情况关闭
rcnn_inference_post_process = True

## 如果设为 True, 则 Fast RCNN 输出时直接输出固定 tensor，为了性能考虑，
# 设置时需要检查全局文件，该内容是否被注释，
for_calculate_flops = False
