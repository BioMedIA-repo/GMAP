import copy
import logging
import os
from pathlib import Path
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from filter import EpochProxySelector
from utils.pixel_image import generate_fusion_image_batch_smooth, MultiClassGaussianUpdater

from utils.utils import ema_update, get_mid_image, monitor_segmentation
from val import evaluate

logger = logging.getLogger('main_logger')


def sup_train(sup_model, supervised_dataloader, source_val_dataloader, target_val_dataloader,
              sup_opt, contrastive_loss, seg_loss, num_batch_per_epoch, checkpoint_dir,
              sup_scheduler, args, device):
    global_step = 0
    no_improve_count = 0
    contra_weight = args.contra_weight
    best_val_score = 0
    best_T_val_score = 0
    max_epoch = 0
    sup_model.train()
    checkpoint_name = args.checkpoint_name
    sup_checkpoint_dir = os.path.join(checkpoint_dir, f'{args.stage}', checkpoint_name)
    train_writer = SummaryWriter(
        log_dir=f'../tensorboard/logs/{args.mode}/{args.stage}/{checkpoint_name}')

    source_updater = MultiClassGaussianUpdater(feature_dim=1, num_classes=5, K=args.N)
    target_updater = MultiClassGaussianUpdater(feature_dim=1, num_classes=5, K=args.N)
    for epoch in range(1, args.sup_epochs + 1):
        epoch_loss = 0
        batch_count = 0
        print('checkpoint:', checkpoint_name, args.mode)
        progress_bar = tqdm(total=num_batch_per_epoch, desc=f"Epoch {epoch}/{args.sup_epochs}",
                            unit="batch")
        for supervised_batch in supervised_dataloader:
            torch.cuda.empty_cache()
            ss_batch, st_batch, label = supervised_batch['s'].to(device), \
                supervised_batch['s2t'].to(device), supervised_batch['label'].to(device)
            mid_batch = get_mid_image(ss_batch, st_batch)
            if epoch == 1:
                source_updater.update(ss_batch, label)
                target_updater.update(st_batch, label)
            source_mid_batch, target_mid_batch = generate_fusion_image_batch_smooth(ss_batch, st_batch, label,
                                                                                    source_updater, target_updater)
          
            sup_sourcelike_features = sup_model.encoder(ss_batch)
            sup_targetlike_features = sup_model.encoder(st_batch)
            sup_mid_features = sup_model.encoder(mid_batch)
            sup_source_mid_features = sup_model.encoder(source_mid_batch)
            sup_target_mid_features = sup_model.encoder(target_mid_batch)

            sup_sourcelike_proj = sup_model.projection(sup_sourcelike_features[-1])
            sup_targetlike_proj = sup_model.projection(sup_targetlike_features[-1])
            sup_mid_proj = sup_model.projection(sup_mid_features[-1])
            sup_source_mid_proj = sup_model.projection(sup_source_mid_features[-1])
            sup_target_mid_proj = sup_model.projection(sup_target_mid_features[-1])

            contrastive_pairs = [
                (sup_sourcelike_proj, sup_mid_proj),
                (sup_targetlike_proj, sup_mid_proj),
                (sup_sourcelike_proj, sup_source_mid_proj),
                (sup_targetlike_proj, sup_target_mid_proj),
                (sup_mid_proj, sup_source_mid_proj),
                (sup_mid_proj, sup_target_mid_proj),
            ]

            contra_loss = sum(contrastive_loss(pair[0], pair[1]) for pair in contrastive_pairs)

            s_pred = sup_model.decoder(*sup_sourcelike_features)
            t_pred = sup_model.decoder(*sup_targetlike_features)
            m_pred = sup_model.decoder(*sup_mid_features)

            s_seg_loss = seg_loss(s_pred, label)
            t_seg_loss = seg_loss(t_pred, label)
            m_seg_loss = seg_loss(m_pred, label)

            total_seg_loss = (t_seg_loss + s_seg_loss + m_seg_loss) / 3

            total_loss = total_seg_loss + contra_weight * contra_loss

            progress_bar.set_postfix(cl=contra_weight * contra_loss.item(), seg=total_seg_loss.item())
            progress_bar.update(1)
            train_writer.add_scalar('SupTrain/Loss', total_loss.item(), global_step)
            train_writer.add_scalar('SupTrain/Seg', total_seg_loss.item(), global_step)
            train_writer.add_scalar('SupTrain/Contra', contra_weight * contra_loss.item(), global_step)
           
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sup_model.parameters(), 1.0)
            sup_opt.step()
            sup_opt.zero_grad()

            global_step += 1
            epoch_loss += total_loss.item()
            batch_count += 1
        sup_scheduler.step()
        current_lr = sup_opt.param_groups[0]['lr']  
        logger.info(f"Epoch [{epoch}], Current Learning Rate: {current_lr}")
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_writer.add_scalar('SupTrain/AvgEpochLoss', avg_epoch_loss, epoch)
        progress_bar.close()

        source_val_score = evaluate(source_val_dataloader, sup_model, num_classes=args.classes)
        target_val_score = evaluate(target_val_dataloader, sup_model, num_classes=args.classes)
        train_writer.add_scalar('Val/source_val_score', source_val_score, epoch)
        train_writer.add_scalar('Val/target_val_score', target_val_score, epoch)

        Path(sup_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        no_improve_count += 1
        val_score = (source_val_score + target_val_score) / 2
        if val_score > best_val_score:
            best_val_score = val_score
            max_epoch = epoch
            logger.info(f'Max_score is {best_val_score} at epoch{epoch}!')
            save_best_path = os.path.join(sup_checkpoint_dir,
                                          '{}_best_model_{}_{}.pth'.format(args.model_type, epoch,
                                                                           f"{best_val_score:.6f}"))
            best_path = os.path.join(sup_checkpoint_dir, '{}_best_model.pth'.format(args.model_type))
            torch.save(sup_model.state_dict(), save_best_path)
            torch.save(sup_model.state_dict(), best_path)
            no_improve_count = 0  
        
        logger.info('best_val_score: %f, epoch: %d', best_val_score, max_epoch)
    train_writer.close()


def unsup_train(unsup_model, mixed_dataloader, supervised_dataloader, source_val_dataloader,
                target_val_dataloader, opt, contrastive_loss, seg_loss, checkpoint_dir, scheduler, args, device):
    checkpoint_name = args.checkpoint_name
    train_writer = SummaryWriter(
        log_dir=f'../tensorboard/logs/{args.mode}/{args.stage}/{checkpoint_name}/contra_weight_{args.contra_weight}_{args.N}')

    unsup_checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name)
    global_step = 0
    max_epoch = 0
    no_improve_count = 0
    best_val_score = 70
    contra_weight = args.contra_weight
    is_update = False
    ema_model = copy.deepcopy(unsup_model)
    unsup_model.train()
    best_model = copy.deepcopy(unsup_model)
    threshold = args.threshold 
    selector = EpochProxySelector(num_classes=5, args=args)
    start_epoch = args.start_epoch
    warm_step = 0

    source_updater = MultiClassGaussianUpdater(feature_dim=1, num_classes=5, K=args.N)
    target_updater = MultiClassGaussianUpdater(feature_dim=1, num_classes=5, K=args.N)
    initialize_up(source_updater, target_updater, device, supervised_dataloader, args)
    for epoch in range(1, args.unsup_epochs + 1):
        epoch_loss = 0
        batch_count = 0
        selector.reset_epoch()
        print('checkpoint:', checkpoint_name, args.mode)
        progress_bar = tqdm(total=len(mixed_dataloader), desc=f"Epoch {epoch}/{args.unsup_epochs}",
                            unit="batch")
        for i_batch, (supervised_batch, unsupervised_batch) in enumerate(mixed_dataloader):
            torch.cuda.empty_cache()
            ss_batch, st_batch, label = supervised_batch['s'].to(device), \
                supervised_batch['s2t'].to(device), supervised_batch['label'].to(device)
            us_batch, ut_batch = unsupervised_batch['s2t'].to(device), \
                unsupervised_batch['s'].to(device)
            len_sup = ss_batch.size(0)
            sm_batch = get_mid_image(ss_batch, st_batch)
            um_batch = get_mid_image(us_batch, ut_batch)

            if epoch == start_epoch: warm_step = global_step
            with torch.no_grad():
                ema_model.eval()
                s_epreds = ema_model(us_batch)
                m_epreds = ema_model(um_batch)  # [B, C, H, W]
                t_epreds = ema_model(ut_batch)
                preds = [s_epreds, m_epreds, t_epreds]
                if epoch > start_epoch:
                    step = global_step - warm_step
                    selector.update_topk_ratio(step)
                    selector.update_score_distribution(preds)
                    selector.update_selection_threshold(50)
                    preds, pixel_weights, filtered_tensors = selector.filter_batch(
                        preds, us_batch, um_batch, ut_batch) 
           
                    us_batch, um_batch, ut_batch = filtered_tensors
                    mean_pred = torch.stack(preds, dim=0).mean(dim=0).argmax(dim=1)
                    pseudo_label = one_hot(mean_pred, num_classes=5).permute(0, 3, 1, 2)
                    pixel_weights = pixel_weights.unsqueeze(1).expand_as(pseudo_label)

                    train_writer.add_scalar('UnsupTrain/k', pseudo_label.shape[0], global_step)

                else:
                    pseudo_label = torch.mean(torch.stack(preds, dim=0), dim=0)
                    pixel_weights = torch.ones_like(pseudo_label)
            ema_model.train()

            if is_update:
                target_updater.update(ut_batch, pseudo_label)
            is_update = False
            usc_batch, utc_batch = generate_fusion_image_batch_smooth(us_batch, ut_batch, pseudo_label,
                                                                      source_updater, target_updater)

            real_batch = torch.cat([ss_batch, ut_batch])
            fake_batch = torch.cat([st_batch, us_batch])
            m_batch = torch.cat([sm_batch, um_batch])
            real_features = unsup_model.encoder(real_batch)
            fake_features = unsup_model.encoder(fake_batch)
            mid_features = unsup_model.encoder(m_batch)
            sourcelike_centroi_mid_features = unsup_model.encoder(usc_batch)
            targetlike_centroi_mid_features = unsup_model.encoder(utc_batch)

            real_proj = unsup_model.projection(real_features[-1])
            fake_proj = unsup_model.projection(fake_features[-1])
            mid_proj = unsup_model.projection(mid_features[-1])
            sourcelike_centroi_mid_proj = unsup_model.projection(sourcelike_centroi_mid_features[-1])
            targetlike_centroi_mid_proj = unsup_model.projection(targetlike_centroi_mid_features[-1])

            contrastive_pairs = [
                (real_proj[len_sup:], mid_proj[len_sup:]),
                (fake_proj[len_sup:], mid_proj[len_sup:]),
                (real_proj[len_sup:], targetlike_centroi_mid_proj),
                (fake_proj[len_sup:], sourcelike_centroi_mid_proj),
                (mid_proj[len_sup:], sourcelike_centroi_mid_proj),
                (mid_proj[len_sup:], targetlike_centroi_mid_proj),
            ]

            contra_loss = sum(contrastive_loss(pair[0], pair[1]) for pair in contrastive_pairs)

            r_pred = unsup_model.decoder(*real_features)
            f_pred = unsup_model.decoder(*fake_features)
            m_pred = unsup_model.decoder(*mid_features)

            r_consistency_loss = seg_loss(r_pred[len_sup:], pseudo_label.detach(), pixel_weights=pixel_weights)
            f_consistency_loss = seg_loss(f_pred[len_sup:], pseudo_label.detach(), pixel_weights=pixel_weights)
            m_consistency_loss = seg_loss(m_pred[len_sup:], pseudo_label.detach(), pixel_weights=pixel_weights)

            r_seg_loss = seg_loss(r_pred[:len_sup], label) + r_consistency_loss
            f_seg_loss = seg_loss(f_pred[:len_sup], label) + f_consistency_loss
            m_seg_loss = seg_loss(m_pred[:len_sup], label) + m_consistency_loss

            total_seg_loss = (r_seg_loss + f_seg_loss + m_seg_loss) / 3
            total_loss = total_seg_loss + contra_weight * contra_loss
            progress_bar.set_postfix(cl=contra_weight * contra_loss.item(), seg=total_seg_loss.item(),
                                     k=pseudo_label.shape[0])
            progress_bar.update(1)
            train_writer.add_scalar('UnsupTrain/Loss', total_loss.item(), global_step)
            train_writer.add_scalar('UnsupTrain/Contra', contra_weight * contra_loss.item(), global_step)
            train_writer.add_scalar('UnsupTrain/Seg', total_seg_loss.item(), global_step)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(unsup_model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            global_step += 1
            epoch_loss += total_loss.item()
            batch_count += 1

        scheduler.step()
        current_lr = opt.param_groups[0]['lr'] 
        logger.info(f"Epoch [{epoch}], Current Learning Rate: {current_lr}")
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_writer.add_scalar('UnsupTrain/AvgEpochLoss', avg_epoch_loss, epoch)
        progress_bar.close()

        source_val_score = evaluate(source_val_dataloader, unsup_model, num_classes=args.classes)
        target_val_score = evaluate(target_val_dataloader, unsup_model, num_classes=args.classes)
        train_writer.add_scalar('Val/source_val_score', source_val_score, epoch)
        train_writer.add_scalar('Val/target_val_score', target_val_score, epoch)

        Path(unsup_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        no_improve_count += 1

        if target_val_score > best_val_score:
            best_val_score = target_val_score
            max_epoch = epoch
            logger.info(f"New best score: {target_val_score} at epoch {epoch}!")
            save_best_path = os.path.join(unsup_checkpoint_dir,
                                          f"{args.model_type}_best_model_{epoch}_{target_val_score}.pth")
            best_save_best_path = os.path.join(unsup_checkpoint_dir, f"{args.model_type}_best_model.pth")
            torch.save(unsup_model.state_dict(), save_best_path)
            torch.save(unsup_model.state_dict(), best_save_best_path)
            best_model = copy.deepcopy(unsup_model)
            no_improve_count = 0  
            is_update = True

        if epoch == start_epoch:
            print("updating...")
            ema_model = copy.deepcopy(best_model)
        if epoch > start_epoch:
            print("updating...")
            ema_update(unsup_model, ema_model, alpha=0.9)
        logger.info('best_val_score: %f, epoch: %d,threshold:%f', best_val_score, max_epoch, threshold)
    train_writer.close()


def initialize_up(source_updater, target_updater, device, supervised_dataloader, args):
    print("Initializing updaters...")
    for supervised_batch in tqdm(supervised_dataloader, desc="Processing batches", unit="batch"):
        torch.cuda.empty_cache()
        sup_sourcelike_batch, sup_targetlike_batch, label = supervised_batch['s'].to(device), \
            supervised_batch['s2t'].to(device), supervised_batch['label'].to(device)
        source_updater.update(sup_sourcelike_batch, label)
        target_updater.update(sup_targetlike_batch, label)
