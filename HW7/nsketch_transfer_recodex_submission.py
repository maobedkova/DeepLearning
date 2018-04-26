# coding=utf-8

source_1 = """#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import nets.nasnet.nasnet


class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data[\"images\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
            len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
                len(self._images))
            return True
        return False


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=True)
            self.nasnet_saver = tf.train.Saver()

            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            # Training
            with tf.variable_scope(\"output_layer\"):
                features = tf.layers.dense(features, 224, activation=tf.nn.relu)
                output_layer = tf.layers.dense(features, self.LABELS, activation=None)
                self.predictions = tf.argmax(output_layer, axis=1)
                self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer)

            global_step = tf.train.create_global_step()

            start_learn_rate = 0.001
            final_learn_rate = 0.0005
            batches_per_epoch = len(train.labels) // args.batch_size
            decay_rate = np.power(final_learn_rate / start_learn_rate, 1 / 2)#args.epochs - 1))
            self.learning_rate = tf.train.exponential_decay(start_learn_rate,
                                                            global_step,
                                                            batches_per_epoch,
                                                            decay_rate, staircase=True)

            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=\"output_layer\")

            updated_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updated_ops):
                self.training = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                    var_list=train_vars,
                                                                                    global_step=global_step,
                                                                                    name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
                         {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.images: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)

    def save(self, model):
        self.saver.save(self.session, model)

    def restore(self, model):
        self.saver.restore(self.session, model)


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=100, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=5, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    #parser.add_argument(\"--mode\", type=str, help=\"Mode (train, test)\")
    #parser.add_argument(\"--epoch\", type=int, help=\"Iteration of train data\")
    #parser.add_argument(\"--start_batch\", type=int, help=\"Checkpoint in batches\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value)
                  for key, value in sorted(vars(args).items()))).replace(\"/\", \"-\")
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset(\"nsketch-train.npz\")
    dev = Dataset(\"nsketch-dev.npz\", shuffle_batches=False)
    test = Dataset(\"nsketch-test.npz\", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # # Train
    # if args.mode == \"train\":
        # if args.epoch != 1 and args.start_batch != 0:
            # network.restore(\"./models/model_{}e_{}b\".format(args.epoch, args.start_batch))
        # for i in range(args.epochs):
            # count = 0
            # while not train.epoch_finished():
                # images, labels = train.next_batch(args.batch_size)
                # count += len(images)
                # if count <= args.start_batch and (args.epoch != 1 and args.start_batch != 0):
                    # continue
                # network.train_batch(images, labels)
                # print(count, \"/\", len(train.images))
                # if count % 100 == 0:
                    # network.save(\"./models/model_{}e_{}b\".format(args.epoch, count))
                    # print(\"Model {}e {}b saved\".format(args.epoch, count))
                    # break
            # accuracy = network.evaluate(\"dev\", dev, args.batch_size)
            # print(\"Acc:\", accuracy)
    # else:
        # network.restore(\"./models/model_{}e_{}b\".format(args.epoch, args.start_batch))

    for i in range(args.epochs):
        print(\"Epoch\", i)
        total = 0
        while not train.epoch_finished():
            images, labels = train.next_batch(args.batch_size)
            network.train_batch(images, labels)
            total += len(images)
            print(total, \"/\", len(train.images))
    
        accuracy = network.evaluate(\"dev\", dev, args.batch_size)
        print(\"Acc:\", accuracy)

    # Predict test data
    with open(\"nsketch_transfer_test.txt\", \"w\") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;2Yfv@m&BKjJ-`TO1<TzKXru$wgN9Lr7UQPdjUL`lzHIm0sp@M0`=l)KZ?SI)2P%eB2!_LEbMQ?-Nm#>jc3rV(Nrge(^R-4m|n!McaQg0CfDjqbq0yNfIU4UCuR_$USbS(_{mlcsgZSKZ6iXD@>G;k{}cA%ZDLNABSCgRvQD_&_In~J8m?dMxd+h951Q4s>13vyQSbH!_IG81woO(<y)6x*`)X>*B!p8w4B7Cp#rBnrN#H;NiZAYv1pp~~@Tv2{F}JH~laY#@!E6aPav!24vABI4I_puYyI5GoE9mjo9X?vQX`4KpZ}obl;w&Pdu!1p)BE9NPl79@mqBHNZBvJt~Bn6aNjgdrdGAD|5ehm*uRC%zK@BE2AvWn{Lr5Y)>k(<3`ma;@fMaSm54X%+kSnUx4kWrvO@qswtXia;qge0O!Cy52`lmG?FNB|-FIsKX&^4YV2{#>o4N@h0k^pe1Gl^YiWGH|1JVpp%JU%LUz6uq1tn<G-2Rd+Zzb4e01)SFyjb#SWS%bt51on=YiaG|}3odkT$O&;@mgv`Zz5Bp92j>MiSRiz13;CT|ky9*H3q1&!6q}MA4z$&#8;WB(biM5C}la2*lIJICbZu?{I$T~1J)Y=<9$pE(cea#ipO*K(rd^&Jvj-p1J>S5>o6G;-XOvaQ){=>d3z4B#?@+EIv^<WVRu=oDsa9g3!W7<@*K}&{obH6DME1n%gpC4cn%Vm2bK<=&o-Bi`ovOT&#zu(v<sON>3$8`*DWQF)A9tU0OePSPCPgh5MQWt}6^H=sw)><l`5Cb@a-F4&zdAx472rVpHP+Vy7(el6ObN^8hX=mg(vnYHHGaP6_4Ao^_QDoLobJ1a#+&11t+h2UDh^4tMu=R+prCwfY=P@H*yu<+08NEvzp3q}H4@ajww9%`wdAi<VMqO78jEe&Qr)PDZJTEs_8juoE(T~L#VAClxB6a1z#m6`SuTv7BxyLH|P2dEY!=>^_v4&yzkr@|Op3#ok&NnF-`k;&62<rILoI@lkWFR*|)6_0)0(Bj|*}Kel)O1s)G18<j@SVruP4G<dH`mIpR_w6-moZm4bCV$=-tTjI;To(MqhuIZfK76bq<W<;6!}>2m3~WQZGyVn9!KIchJy+{^!BOhFR8{N$3QfE)|>YnWuW-pKHYW_eQL2qQ-S31Mk$9#QR}~5RtueH;BPhA%HJSeA0RbXn!TE}Z6c4=;4BC>x#4Z3Rt_d?r7~?6?j3UEr^yo|47~^Y?8g6jd~Z7V>q(!UuW-~81S~zL_xo25SHcZ3O=AXohsOP|o*wf4WlJ++eOyDa^;wgRS3^J_DF7j>Xa}G$|KwPKSt0k4*LS5gl*Eg=l{kG(6`>B%6MtAHKn8I<a-9gSdxqsZAlHpcCj!y6bK1kZU||Ubu%fsmhp%Y#*;0Wb6nQ>(5WD`g-Mcyc-bBb10V7^OOa7R@!(Z<~>ElLw{zFnjAJ*?@3dFwlF7jSTt{`#rk+puC_nBfJ7RMZLL2{eJfdl}qfGXR$OQHA#R6mNnMZ461;0T1#ZPfO(V(S~c7Q<Vp<iP-~L&H;o=nHWYgh$=P*m?xXBl@ahttAUqkG`NQ-ad*VS$!DQ`Bpr%nT*+|us7+uXU_@+y%>=8yUWZx|NbjewH-g+EC-}H^fXV2(y;+Rs2zlz)qVT3oZ?CV2(CkF+_57W$6a(Z?`wDW4Jd?*l$8$H31{GDs4hkODM~Qm@nBR7%CFm{F8yK4d4n{1C<lSgo3?(yIiDP-Oe>x&^<Ae`dhQc>+T=+bi~<z3n3N5PP0YI#GrGY7nsW_D(63Za#RIvHVgqk(ZUHIyZBfzMSq#IVb7(1CWh_XV7~Kw(#M+`UVXGb6g-ZZ7DXxZKe-;majSH_D40_Z87XjtTs)%2$eFW!|BlePY7YE2iaCq{f20$)^cb{fv59c|0s9a@0RsO%#rS#loR$6{Drd|_4lGLM@FuIrY@iY`=Ev3{$rhy7=)Q3xTyo{HRjxFnRu^GkMQ?RH;uF8fYiMP9jaigiiQyoC{V`R+f2nl9r@w2>n>b`w3X)us(W3e_;(Ob<0c#n=qjMf5S8)gDX|HeFv>g1?7R04O!d_XW)T6QA}=`Ed6d)p2sZa9@7Y($GS@&{Y51lrfw&4qU<pQ?%po3Dc)TeX-EGTI1e4_z*o*z8|8XwtQ;#Tr+kGa$hqQ*J)_uHQMQ2ZCf)=0D6<t?!(k(t#b|E|qZF4TRU#W7=G>udC)R$+4{l#6KCGPCq-?phwed89}}V?}o?&B3u6|X>(GL5r!4gfNDfY`_=pEA8-~@XRBpTQXKP3om)rzC!Pc7b+E_A8aC@3nK?u)zN!Wg8Kx%rU#YS6=N=0~5MS4#zXF`~YmH+<V)UfvYHk3-i%YJ?k$O(ki~j91Uh5cybw;D6;e0OFRYQ$inWAmnfa>+uxf!p$^W_=c4Z(H9PU$c8r~_MAqWVfpH|c~i01kB56i;N`ILLzrkJ(jjbVFwXq!+5?-zJCkyl%hX2@t<KC@j0kKRbZf!oZXBf<^g$zg^HIJc?6=gc!=iwKqAsqQ`rMyeMks^H9K4f*5-~1wLm_tfDx7MLI0Js9cHR@InYDU?(EgEunnyIMwZplK<au?fiwA3(uZK8RIV%OAAYwDF{8HfUnhf*E1a&j|g=ct*^0jJ?+1u<ny2pgNh|MHw1aJ<ukJ{II0{IK=ixEoJuM}%M}>yaUD))&~8Zs&lIc@IdIYa7hky?Td1lR`jSGiE@3B#_Ab7Cni7W6M){1BJWD3zG`&8uaAZQUzH;Z&bW@*qbdD%GRk>gInLlJG{2#Lcij?xm>J`yDH|h}xr|uRMN#NsFAvbXrG_w@1R%zw3LWNY%Ya)e<#+Y1dJt$2;gUYZyz3YTKr{vfq*4XNs=CJ3}yC@V+Cc;Yr-JV&+XRZdeCz;nJ)Pe2vmTGpY#ApQdM^VS#{Br!=BwxYUWW`qn)R(?2PC#PrWvPxKMSf`NvC>0yxyB%|n4OBd;&hIH-h>Rd<Aq%-vL73K>+*w`6VS`cAA%0^|CwGJ{6D5;ihw)bk?S^E?hOznR*zXRyv~D37Y0K9vod5>g#qO4qT@iot0NHHJ7rl^P+r4Jj&luE`u;m;5Zn8W!l27d;@oJvhb_Y}d_URW`*@SH5iIVCY1<`Z!Sm%KrMb=>te$9gkagx#xIO+2H_y6L)d-4vdo@K#mqhsaj<WBxsxK-&+9496mg9NDBb?lnUX%O4R9_~i)CcZ}hnY=(Xl~l1N0F}zxr-7t(2o$ol$M+w(c{7t(9(yY5c+)GT{z8>#Hop4Dvvg+h`9rJaJcNeJ^$HPU3bdwg1+wF;({+-Cjv^F8?=gv$+&BvI2qN!0f1A+^;YV)@ds$uH;D&)ss{9)F?o61jQ~sZ9BngXhwoXb_BuICyH1Ai0#+KfocKaH-0r7W2Qpx6YP=2Rm$8<K+i8YT9Pca^t-Ca!{Z>g51l*~=`+K}K;S%TNC(I=rsNQ%)We>zkT>|$T!n<uih`+wDow)N99giX&9<WE3z&wbgCW`OIM9+VL<Zn<7)p0DsWOBC^)&!@MN*tl3aQDX1U*y-r^Wg9R00000^v`=OH=f}v00E5_-ZuaMMvQc6vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
