# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from models.parser import arg_parse
from models.unet_onehot_cns_font_attention import UNet
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(_):
    args = arg_parse()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty,
                     cns_encoder_dir=args.cns_encoder_dir, cns_embedding_size=args.cns_embedding_size)
        model.register_session(sess)
        if args.flip_labels:
            model.build_model(is_training=True, inst_norm=args.inst_norm, no_target_source=True)
        else:
            model.build_model(is_training=True, inst_norm=args.inst_norm)
        fine_tune_list = None
        if args.fine_tune:
            ids = args.fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=args.sample_steps, flip_labels=args.flip_labels)


if __name__ == '__main__':
    tf.app.run()
