import torch
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.checkpoint import load_state_dict_with_optional_compatibility
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor
import tqdm


PREDICTION_THRESHOLD = float(getattr(cfg, 'prediction_threshold', 0.9))

if __name__ == '__main__':
    device = "cuda:0"

    net = evspsegnet(cfg).eval()
    net.cuda()

    dataset = EvUAV(cfg, mode='test')

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,collate_fn=dataset.custom_collate)

    load_state_dict_with_optional_compatibility(
        net,
        torch.load(cfg.model_path, map_location=device),
        p2b_enabled=bool(getattr(net, 'p2b_density_gdsca_enabled', False)),
        p11_enabled=bool(getattr(net, 'p11_local_activity_enabled', False)),
    )
    print('dict load: ',cfg.model_path)

    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    postprocess_stats = postprocessor.new_stats()
    print('postprocessor:', postprocessor.describe())


    pbar = tqdm.tqdm(total=len(test_dataloader), desc='video', unit='video',unit_scale=True,position=0, leave=True)

    evaluter = evalute(cfg)

    for sample,ev in enumerate(test_dataloader):
        with torch.no_grad():
            x = ev['voxel_ev']
            event_frame = ev.get('event_frame')
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()
            idx = ev['idx_label']
            ts = ev_locs[:,3]

            preds, voxel = net(x, event_frame=event_frame)
            preds = preds[p2v_map].squeeze().cpu()
            preds, batch_postprocess_stats = postprocessor.apply(preds, ev['locs'])
            postprocess_stats.merge(batch_postprocess_stats)

            if cfg.eval:
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred']= preds
                evaluter.matches[str(sample)]['seg_gt'] = label
                if cfg.roc:
                    evaluter.roc_update(
                        ts,
                        preds,
                        idx,
                        label.cpu(),
                        ev_locs,
                        thresh=PREDICTION_THRESHOLD,
                    )

        pbar.update(1)

    pbar.close()
    print('postprocess result:', postprocess_stats.summary())

    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou(
            thresh=PREDICTION_THRESHOLD
        )
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy(
            thresh=PREDICTION_THRESHOLD
        )
        if cfg.roc:
            pd, fa = evaluter.cal_roc()
            print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou, seg_acc, pd, fa))
        else:
            print('iou:{},seg_acc:{}'.format(iou, seg_acc))
