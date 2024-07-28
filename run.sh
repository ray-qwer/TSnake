# usage
## python train_net.py <config filename, without ext> \
##              --bs [batch size] --exp-name [exp name] \
##              --model-dir [the place store model]
##              --detect-only

## train detector-bbox
# python train_net.py detector --bs=2 --exp-name finetune-d-bbox --model-dir finetune/d-bbox 

## train snake
# python train_net.py sbd_time_embd --exp-name dilated-only-3-trainable-pos --model-dir sbd/dilated-only-3-trainable-pos --bs 20
# python train_net.py sbd_time_embd --exp-name mosaic-data-aug-0.4 --model-dir sbd/mosaic-data-aug-0.4 --bs 18 --checkpoint data/model/sbd/mosaic-data-aug/64.pth
# python train_net.py sbd_time_embd_aug --exp-name mosaic-data-aug-larger-lr --model-dir sbd/mosaic-data-aug-larger-lr --bs 18 
# python train_net.py sbd_time_embd_aug --exp-name time-emb-output --model-dir sbd/time-emb-output --bs 20 --wandb-id 97z5ntff
# python train_net.py sbd_time_embd --exp-name trainable-time-encode --model-dir sbd/trainable-time-encode --bs 20 --wandb-id rilib5i6
# python test.py sbd_time_embd --checkpoint data/model/sbd/trainable-time-encode/best.pth --dataset sbd_val
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed --model-dir cityscapes/cityscapes_proposed --bs 6
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed-bs8-res768 --model-dir cityscapes/cityscapes_proposed-bs8-res768 --bs 8 --checkpoint data/model/cityscapes/cityscapes_proposed-bs8-res768/latest.pth
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed-bs16-res512 --model-dir cityscapes/cityscapes_proposed-bs16-res512 --bs 16 --checkpoint data/model/cityscapes/cityscapes_proposed-bs16-res512/latest.pth
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed-bs8-res768 --model-dir cityscapes/cityscapes_proposed-bs8-res768-mosaic-aug --bs 8
# python train_net.py cityscapesCoco_proposed_ft --exp-name cityscapes_proposed-bs6-res800-ft --model-dir cityscapes/cityscapes_proposed-bs6-res800-ft --bs 8 --checkpoint data/model/cityscapes/cityscapes_proposed-bs6-res800-ft/latest.pth
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed-bs8-res800-mosaic --model-dir cityscapes/cityscapes_proposed-bs8-res800-mosaic --bs 8 --checkpoint data/model/cityscapes/cityscapes_proposed-bs8-res800-mosaic/latest.pth
# python train_net.py cityscapesCoco_proposed_no_dilated --exp-name cityscapes_proposed-bs8-res800-mosaic-no-dilated --model-dir cityscapes/cityscapes_proposed-bs8-res800-mosaic-no-dilated --bs 8 --checkpoint data/model/cityscapes/cityscapes_proposed-bs8-res800-mosaic-no-dilated/latest.pth
# python train_net.py cityscapesCoco_proposed --exp-name cityscapes_proposed-bs8-res800-mosaic-retrain --model-dir cityscapes/cityscapes_proposed-bs8-res800-mosaic-retrain --bs 8 --checkpoint data/model/cityscapes/cityscapes_proposed-bs8-res800-mosaic-retrain/latest.pth
# python train_net.py cityscapesCoco_proposed_ft --exp-name cityscapesCoco_proposed_ft_sgd-bs6-res800 --model-dir cityscapes/cityscapesCoco_proposed_ft_sgd-bs6-res800-33.4 --bs 6 --checkpoint data/model/cityscapes/cityscapes_proposed-bs8-res768-mosaic-aug/best.pth --type finetune

# python train_net.py coco_proposed --exp-name coco-proposed --model-dir coco/coco-proposed --bs 16 --checkpoint data/model/coco/coco-proposed/latest.pth --wandb-id z6stdmbl
# python train_net.py coco_proposed_ft --exp-name coco-proposed_ft --model-dir coco/coco-proposed_ft --bs 16 --checkpoint data/model/coco/coco-proposed/159.pth --type finetune --wandb-id n13xdc8e
python3 train_net.py cityscapesCoco_proposed_combine --exp-name combine-short-dist-100-rpe-all-layer --model-dir cityscapes/combine-short-dist-100-rpe-all-layer --bs 8 --checkpoint data/model/cityscapes/combine-short-dist-100-rpe-all-layer/latest.pth --wandb-id fbh3fz40

