from aim.utils import DetectionModel, Evaluator, BBoxDrawing

import argparse
import os
import torch

from m2det import build_net
from configs.CC import Config
from data import BaseTransform
from layers.functions import PriorBox, Detect
import torch.backends.cudnn as cudnn
from utils.core import anchors, init_net, get_dataloader, nms_process_for_eval, print_info


class M2Det(DetectionModel):

    def __init__(self, config, trained_model):
        if isinstance(config, str):
            config = Config.fromfile(config)
        self.cfg = config
        self.trained_model = trained_model
        self.anchor_config = anchors(self.cfg)
        print_info('The Anchor info: \n{}'.format(self.anchor_config))
        self.priorbox = PriorBox(self.anchor_config)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if self.cfg.test_cfg.cuda:
                self.priors = self.priors.cuda()
        self.num_classes = cfg.model.m2det_config.num_classes
        self.net = build_net('test',
                             size = self.cfg.model.input_size,
                             config = self.cfg.model.m2det_config)
        init_net(self.net, self.cfg, self.trained_model)
        print_info('===> Finished constructing and loading model',['yellow','bold'])
        
        self.net.eval()
        if self.cfg.test_cfg.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        else:
            self.net = self.net.cpu()

        self.detector = Detect(self.num_classes, self.cfg.loss.bkg_label, self.anchor_config)
        self.transform = BaseTransform(self.cfg.model.input_size, self.cfg.model.rgb_means, (2, 0, 1))

    def __call__(self, *args, **kwargs):
        img = args[0]
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).numpy()
            
        w,h = img.shape[1],img.shape[0]
        scale = torch.Tensor([w,h,w,h])
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.cfg.test_cfg.cuda:
                x = x.cuda()
                scale = scale.cuda()
        out = self.net(x)
        boxes, scores = self.detector.forward(out, self.priors)
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        
        all_boxes = nms_process_for_eval(self.num_classes, scores, boxes, self.cfg, self.cfg.test_cfg.score_threshold, self.cfg.test_cfg.topk)

        results = []
        image_id = kwargs['image_id']
        for i, boxes in enumerate(all_boxes):
            if boxes is None:
                continue
            
            for box in boxes:
                obj = dict()
                obj['image_id'] = image_id
                obj['category_id'] = i # Remove BACKGROUND
                obj['score'] = float(box[4])
                x = float(box[0])
                y = float(box[1])
                w = float(box[2]) - x
                h = float(box[3]) - y
                obj['bbox'] = [x, y, w, h]
                results.append(obj)

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M2Det Testing')
    parser.add_argument('--config', default='configs/m2det320_vgg.py', type=str)
    parser.add_argument('--trained_model', default=None, type=str)
    parser.add_argument('--output_dir', default='output', type=str, help='The directory name of the detection result.')
    parser.add_argument('--output', default='bbox.json', type=str, help='The filename of the detection result.')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = get_dataloader(cfg, 'COCO', 'eval_sets')
    model = M2Det(cfg, args.trained_model)

    evaluator = Evaluator(os.path.join(cfg.COCOroot, 'val.json'), args.output)
    data_loader = torch.utils.data.DataLoader(dataset)
    evaluator.run(model, data_loader)

    bbox_drawing = BBoxDrawing()
    bbox_drawing.run(args.output_dir, os.path.join(cfg.COCOroot, 'val'), os.path.join(cfg.COCOroot, 'val.json'), args.output)
