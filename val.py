import logging
import numpy as np
import torch
from utils import binary


def evaluate(val_dataset, net, num_classes):
    net.eval()
    all_batch_dice = []
    all_batch_asd = []
    logger = logging.getLogger('main_logger')
    logger.info('Validation started.')
    for idx, batch in enumerate(val_dataset):
        xt, xt_labels = batch['s'].cuda(), batch['label'].cuda()
        output = net(xt)
        out = torch.argmax(output, dim=1)

        for ind in range(output.shape[0]):
            batch_dice = []
            batch_asd = []

            out_img = out[ind]
            xt_lab_img = xt_labels[ind].squeeze(0)
            if torch.sum(xt_lab_img) == 0:
                continue

            for i in range(1, num_classes):
                pred = (out_img == i).cpu().numpy()
                gt = (xt_lab_img == i).cpu().numpy()

                dice, jc, hd, asd = calculate_metric_percase(pred, gt)
                batch_dice.append(dice)
                batch_asd.append(asd)

            all_batch_dice.append(batch_dice)
            all_batch_asd.append(batch_asd)

    all_batch_dice = np.array(all_batch_dice)
    all_batch_asd = np.array(all_batch_asd)

    mean_dice = np.mean(np.ma.masked_equal(all_batch_dice, 0), axis=0)
    total_mean_dice = np.mean(mean_dice)
    mean_asd = np.mean(np.ma.masked_equal(all_batch_asd, 0), axis=0)
    total_mean_asd = np.mean(mean_asd)

    logger.info('Per class metrics:')
    logger.info('  Dice : {}'.format(np.round(mean_dice, 3)))
    logger.info('  ASD  : {}'.format(np.round(mean_asd, 3)))
    logger.info('Overall Mean Metrics:')
    logger.info('  Mean Dice: {:.5f}'.format(total_mean_dice))
    logger.info('  Mean ASD : {:.5f}'.format(total_mean_asd))

    return total_mean_dice


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            dice = binary.dc(pred, gt) * 100
            jc = binary.jc(pred, gt)
            hd = binary.hd95(pred, gt)
            asd = binary.asd(pred, gt)
            return dice, jc, hd, asd
        except RuntimeError as e:
            logging.error(f"Error calculating metrics: {e}")
            return 0, 0, 0, 0
    else:
        return 0, 0, 0, 0
