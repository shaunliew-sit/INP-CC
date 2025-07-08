import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.box_ops as box_ops
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw


class Visualizer(object):
    def __init__(self, args):
        if args.vis_dir:
            Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        self.vis_dir = Path(args.vis_dir)
        self.patch_size = args.vision_patch_size

    def visualize_preds(self, images, targets, outputs, vis_threshold=0.1, HOI_CATEGORIES=[]):
        vis_images = images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
        image_masks = images.mask

        for b in range(len(vis_images)):
            img_rgb = vis_images[b]
            img_rgb = img_rgb - img_rgb.min()
            img_rgb = (img_rgb / img_rgb.max()) * 255
            img_pd = Image.fromarray(np.uint8(img_rgb))

            img_id = int(targets[b]["image_id"])
            img_mask = image_masks[b]
            ori_h = int(torch.sum(~img_mask[:, 0]))
            ori_w = int(torch.sum(~img_mask[0, :]))

            # visualize preds
            hoi_scores = outputs["logits_per_hoi"][b].softmax(dim=-1)
            box_scores = outputs["box_scores"][b].sigmoid()
            scores = (hoi_scores * box_scores).detach().cpu()

            boxes = outputs["pred_boxes"][b].detach().cpu()
            pboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])
            oboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, 4:])
            pboxes[:, 0::2] = pboxes[:, 0::2] * ori_w
            pboxes[:, 1::2] = pboxes[:, 1::2] * ori_h
            oboxes[:, 0::2] = oboxes[:, 0::2] * ori_w
            oboxes[:, 1::2] = oboxes[:, 1::2] * ori_h

            keep = torch.nonzero(scores > vis_threshold, as_tuple=True)
            scores = scores[keep].numpy()
            classes = keep[1].numpy()
            pboxes = pboxes[keep[0]].numpy()
            oboxes = oboxes[keep[0]].numpy()

            # draw predictions in descending order
            top_pattern_indices = outputs["top_indices"][b].detach().cpu().numpy()
            top_pattern_indices_str = "".join([str(x) for x in top_pattern_indices])
            
            indices = np.argsort(scores)[::-1]
            for i in indices:
                hoi_id = int(classes[i])
                img_pd = Image.fromarray(np.uint8(img_rgb))
                drawing = ImageDraw.Draw(img_pd)
                top_left = (int(pboxes[i, 0]), int(pboxes[i, 1]))
                bottom_right = (int(pboxes[i, 2]), int(pboxes[i, 3]))
                draw_rectangle(drawing, (top_left, bottom_right), color="blue")

                top_left = (int(oboxes[i, 0]), int(oboxes[i, 1]))
                bottom_right = (int(oboxes[i, 2]), int(oboxes[i, 3]))
                draw_rectangle(drawing, (top_left, bottom_right), color="red")

                dst = Image.new('RGB', (img_pd.width, img_pd.height))
                dst.paste(img_pd, (0, 0))
                # dst.save(self.vis_dir.joinpath(f'image_{img_id}_hoi_{hoi_id}_score_{scores[i]:.2f}.jpg'))
                # dst.save(self.vis_dir.joinpath(f'image_{img_id}_hoi_{HOI_CATEGORIES[hoi_id]}_score_{scores[i]:.2f}.jpg')) 
                # also add the top pattern indices
                # import os; os.mkdir(self.vis_dir.joinpath(f'top_pattern_{top_pattern_indices_str}'), exist_ok=True)TypeError: 'exist_ok' is an invalid keyword argument for mkdir()
                Path(self.vis_dir.joinpath(f'ysfffvis/top_pattern_{top_pattern_indices_str}')).mkdir(parents=True, exist_ok=True)
                dst.save(self.vis_dir.joinpath(f'ysfffvis/top_pattern_{top_pattern_indices_str}/image_{img_id}_hoi_{HOI_CATEGORIES[hoi_id]}_score_{scores[i]:.2f}_pattern_{top_pattern_indices_str}.jpg'))


    def visualize_attention(self, images, targets, outputs, vis_threshold=0.1, HOI_CATEGORIES=[]):
        vis_images = images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
        image_masks = images.mask
        bs, h, w, _ = vis_images.shape

        pred_info = []
        for b in range(bs):
            img_rgb = vis_images[b]
            img_rgb = img_rgb - img_rgb.min()
            img_rgb = (img_rgb / img_rgb.max()) * 255
            img_pd = Image.fromarray(np.uint8(img_rgb))

            img_id = int(targets[b]["image_id"])
            img_mask = image_masks[b]
            ori_h = int(torch.sum(~img_mask[:, 0]))
            ori_w = int(torch.sum(~img_mask[0, :]))

            # visualize preds
            hoi_scores = outputs["logits_per_hoi"][b].softmax(dim=-1)
            box_scores = outputs["box_scores"][b].sigmoid()
            scores = (hoi_scores * box_scores).detach().cpu()

            boxes = outputs["pred_boxes"][b].detach().cpu()
            pboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])
            oboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, 4:])
            pboxes[:, 0::2] = pboxes[:, 0::2] * ori_w
            pboxes[:, 1::2] = pboxes[:, 1::2] * ori_h
            oboxes[:, 0::2] = oboxes[:, 0::2] * ori_w
            oboxes[:, 1::2] = oboxes[:, 1::2] * ori_h

            keep = torch.nonzero(scores > vis_threshold, as_tuple=True)
            scores = scores[keep].numpy()
            classes = keep[1].numpy()
            pboxes = pboxes[keep[0]].numpy()
            oboxes = oboxes[keep[0]].numpy()

            # draw predictions in descending order
            indices = np.argsort(scores)[::-1]
            cur_pred_info = []
            for i in indices:
                hoi_id = int(classes[i])
                img_pd = Image.fromarray(np.uint8(img_rgb))
                # save the original image
                img_pd.save(self.vis_dir.joinpath(f'image_{img_id}_origin.jpg'))
                drawing = ImageDraw.Draw(img_pd)
                top_left = (int(pboxes[i, 0]), int(pboxes[i, 1]))
                bottom_right = (int(pboxes[i, 2]), int(pboxes[i, 3]))
                draw_rectangle(drawing, (top_left, bottom_right), color="blue")

                top_left = (int(oboxes[i, 0]), int(oboxes[i, 1]))
                bottom_right = (int(oboxes[i, 2]), int(oboxes[i, 3]))
                draw_rectangle(drawing, (top_left, bottom_right), color="red")

                dst = Image.new('RGB', (img_pd.width, img_pd.height))
                dst.paste(img_pd, (0, 0))
                dst.save(self.vis_dir.joinpath(f'image_{img_id}_hoi_{HOI_CATEGORIES[hoi_id]}_score_{scores[i]:.2f}.jpg'))

                # visualize attention maps
                attn_map = outputs["attn_maps"][b]
                token_id = keep[0][i]
                attn = attn_map[token_id, :].view(1, 224 // self.patch_size, 224 // self.patch_size)
                attn = attn - attn.min()
                attn = attn / attn.max()

                attn_tosave = attn.clone().cpu()  # torch.Size([1, 14, 14])
                encoded_image_feat_tosave = outputs["encoded_image_feat"][b].view(224 // self.patch_size, 224 // self.patch_size, -1).cpu() # torch.Size([14, 14, 768])
                decoded_image_feat_tosave = outputs["decoded_image_feat"][b].view(224 // self.patch_size, 224 // self.patch_size, -1).cpu() # torch.Size([14, 14, 768])
                # attn = F.interpolate(attn.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0][0].detach().cpu().numpy()
                # interpolate according to the original image size
                attn = F.interpolate(attn.unsqueeze(0), size=(ori_h, ori_w), mode="nearest")[0][0].detach().cpu().numpy()
                plt.imsave(self.vis_dir.joinpath(f'image_{img_id}_hoi_{HOI_CATEGORIES[hoi_id]}_score_{scores[i]:.2f}_attn.jpg'), arr=attn, format='jpg')

                import cv2
                # crop img_rgb using origin_h, origin_w
                img_rgb = img_rgb[:ori_h, :ori_w, :]  # (448, 672, 3), range: 0-255, <class 'numpy.ndarray'>
                attn = attn * 255  # (448, 672), range: 0-255, <class 'numpy.ndarray'>
                attn = np.uint8(attn)  # Convert attention map to uint8
                # attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
                attn = cv2.applyColorMap(attn, cv2.COLORMAP_HOT)
                attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
                attn = cv2.resize(attn, (ori_w, ori_h))
                img_rgb = np.uint8(img_rgb)  # If it's not already uint8, cast it to uint8
                # overlay attn on img_rgb
                overlay = cv2.addWeighted(img_rgb, 0.5, attn, 0.5, 0)
                # save overlay
                cv2.imwrite(str(self.vis_dir.joinpath(f'image_{img_id}_hoi_{HOI_CATEGORIES[hoi_id]}_score_{scores[i]:.2f}_overlay.jpg')), overlay)
                
                
                cur_pred_info.append((img_id, HOI_CATEGORIES[hoi_id], hoi_id, scores[i], attn_tosave, encoded_image_feat_tosave, decoded_image_feat_tosave))
            pred_info.append(cur_pred_info)
        return pred_info


def draw_rectangle(draw, coordinates, color, width=4):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)