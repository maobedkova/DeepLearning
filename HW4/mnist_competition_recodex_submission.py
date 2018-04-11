# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.

            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # I
            features = tf.layers.conv2d(inputs=self.images,
                                        filters=32,
                                        strides=[2, 2],
                                        kernel_size=[10, 10],
                                        padding=\"same\",
                                        activation=tf.nn.relu)

            features = tf.nn.relu(tf.layers.batch_normalization(features, training=self.is_training))

            features = tf.layers.max_pooling2d(inputs=features,
                                               pool_size=[2, 2],
                                               strides=2)

            # II
            features = tf.layers.conv2d(inputs=features,
                                        filters=32,
                                        strides=[2, 2],
                                        kernel_size=[5, 5],
                                        padding=\"same\",
                                        activation=tf.nn.relu)

            features = tf.nn.relu(tf.layers.batch_normalization(features, training=self.is_training))

            features = tf.layers.max_pooling2d(inputs=features,
                                               pool_size=[2, 2],
                                               strides=2)

            features = tf.layers.flatten(features, name=\"flatten\")

            dropout = 0.3
            features = tf.layers.dropout(features,
                                         rate=dropout,
                                         training=self.is_training,
                                         name=\"dropout_layer\")


            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()

            start_learn_rate = 0.005
            final_learn_rate = 0.000005
            batches_per_epoch = mnist.train.num_examples // args.batch_size
            decay_rate = np.power(final_learn_rate / start_learn_rate, 1 / (args.epochs - 1))
            self.learning_rate = tf.train.exponential_decay(start_learn_rate,
                                                            global_step,
                                                            batches_per_epoch,
                                                            decay_rate, staircase=True)

            updated_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updated_ops):
                self.training = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,
                                                                                    global_step=global_step,
                                                                                    name=\"training\")



            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
                         {self.images: images, self.labels: labels, self.is_training: True})


    def evaluate(self, dataset, images, labels):
        return self.session.run([self.accuracy, self.summaries[dataset]],
                                {self.images: images, self.labels: labels, self.is_training: False})

    def predict(self, dataset, images):
        return self.session.run([self.predictions, self.summaries[dataset]],
                                {self.images: images, self.labels: [0] * len(images), self.is_training: False})


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=15, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42,
                                            source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        evl = network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)[0]
        # print(\"{0:.2f}\".format(evl * 100))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images

    test_labels = network.predict(\"test\", mnist.test.images)[0]

    with open(\"mnist_competition_test.txt\", \"w\") as test_file:
        for label in test_labels:
            print(label, file=test_file)

"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;B_w+U0nbf13gV=7Yq_8Jt(?mHZYog4pqV6G^cyK9Szyu=and!)1{nG33dLhx%)xq>_3>kdU~a)(0^-?KQi8Nnlbf)8EbDt5U9STBxaRafP_ak9`i;tKO!3gJ&#<m<pHT5!H=)dDcm!M9ozOA{XDpT8LSQmd$^@Zt8J&z!}&M!?{IYPf^5A%d+1-oXG5Az%%Wm}eQ*PwYp%!SF#T>wvA>I;3vEn`r(+&;A)9n_*_lIqganW7*-Q6NX<+w|M?zZ77>T#3eJB81Q@1=6W@rvT-$CFb-EarqSA>wKSF2%kg5N&Xl(|0GL^Pl<=?5R9lxwK>=9BJI(x4fn7FLp%qm<tR?4f=<YU@v(KZ<?o&c_{4j79n^dzAQZotK(ZYc_~?$ch#59j^qa?Ele!-h!oQgUpkV<#CSp(9ZV^-R_D$S<#(A*7vC{D90^VhUFr>TKJ_zv}rl}4<CDluI(O_cq}Z#7t=_jun%j+y!mD(yH@d4IGeMO&o=Ku@_DJP!yVi$Cs<Cin>TK+09c{?l6YFZ9wtQ*fpXG0FpY&FR@5<Aub;fjDaf9((E4dVghE}=YzMC?ndo8=G%mg$-0nT;o9z2c>!cwuLfw@!L1#<211Z3>vOW&UTY#QSg4Ac$JVQZbk&en2>h8+VO!KnvQMU8?fGx*`JP@yoY&bZ^lj?xf4eWwyQ+<kqN!CG|{wYe)U>|N#j==&V^hUFtiyG58itHt{ezs_fwMrP#_+tBNxX7V`K0N#58YI7jhUI8IUaheP2UiNA;liou`<;#W0tRPynFUX%$D*DPVKdA|WTsKPz^CBCHjCE|025Vfso186g`hFM5r&9Qt~U6k54E0#pH!D_EjRto(w|$;<`#ZwxrTNciB_?%vNX5knE3Os!-~hg3rFv7qwR;I2=C7t4xvLk6^7v%Pq;SOREsH1Cpr0er(N$cGx}`repm&7Q$ESS{=V+!w$^_71fTiP-}5E|iA+qX(4Ok@-;m~Bo$FOL$f$0paoxxMN*7BQzhY49bow&Eau0$<NTFbRkX#(pZO<{9LBh&omR>n>{)YhtEWq}gA-^%a07{F+Ys}!#U%0NdjSmM`U?fKLHh^j%T|d{}QZ)o)<r@#2sn5Fx4U~9$ebg~mKDZG%^zuYRipeaF$%9|fW&=ZReTH6fx^_3y4k;>9e5;PrD#TQm)jvv#e3ONM0nhsc8wKPQ$BBuYHR)q7LdG&;5BzS6<f6G`1wel<iWn}|@p`eT^ZUm>Xwp;V05BjIDt-<qG1-3*>=1>~%I|^Jsn%%<P;d8r1a}SUf9d40btMQF5BibRHhJ~0oyLF<p!F-YaZjF4%!MMoLB<cnfdpExoN|7tuHyf(;*49L(qD<;@ICP~D!W1Ca;v4M{EE9cA>el%{@%jHh8u2)6qjvV@_2ZZYBY8_Zh@6k#Fy}<f~aWVFj8qIx_^QrqCKYtH@HCvVL{Qdj{0593Tkhola=RnV&^ye%3q3m!FvZ@P}raqfkEb=Y>hNh#d7W)9Z{8_6@;>|Du|{becMC;L=8!-po;NUvkkOh0p5%AXOsN56hp!PkF4B5a}oxt0od-<G*^?1*<o76iUWG2y&a3Z^c~5zGru~sd%s}_IASy0hNFY?2uw_1U=(I>S1i6S$@4Z^V+=9{_e7g?x{>{5ozhXng#(m-|BF(2FPIDir=U8cYP^K>It>=Rm7x#{?%>I3QrV7juiu^UqfHF5+pJzKB0d1ShqqNvGxtsl`aMXf#IRJwo0~U<{CBgr!&I0V4xT9(jToq}r&^UT?cGOhMAFY9j}?)w7bE<-K@?E`*%p`rm)EI&yP2=+U_DD}(hXHorOgfs1sj@z98_(2(sW?tbjG*9qVb#vMFJAl6>#AO3V}seJo>gGLAbo;Y}EzM!d;g3ZC$R7_XR$e)aZYXhfInlglhhh&;`#~carY7uK)s`P~35M_tX0gPI0~F%9#VsO6w%HSKT@i5QpHMAFd#svj_z|!80$Mo3Hy8|7@1OC!SMYVci7h`){DRy@vCX!un(bz3e7S&S*Cqv{%ovxFhtcPQU02RdTVtJnKXk%EJz@TxzCFLIo@gP61DtL1gJ^**p4X7g7bh39n5U$t_A;34PKMs!!#(cq+GoAD$u?P$&0xEO6{JwPE^++(+uoKrJe1>A-Nt#v$czFtGjt3Ff3mu1K8>dDQ9KE6(@5ZYjyQ<M^Pt{%#_ftBS#=H%p-V5@N`b_D8He$b-af=^TduFG9Ja-(lz*uG42akRXsOA>bFrl;X>Ta{8MX&t$oj9Aa<mvpPQx#}RE5Eu`pZD~u`Pb?^NcXT(JzXWoMkCb&D9vN~4GI`uCq=(CLob!{p>rO}ZW@(4N$$uj>v#zmCRC;YogX%Il+EI|<*yZ0P-!LP3jA0Q~{uI5v4XSzP7`EdcF4LoU>Cl@_9M$|nfU+A4AomH>g;ig|l#ywXChl1ABtu<Nqt8l5~Z@6)gC%P=U%`WtyxcaMHM1kmPE%v%N1UKo!Ifrs9J+4mEQ#hw`M4hmkuZ^Wkh_o|(H%N`W_F=@fdyy8nvT9$|dK?OO&O3Hf&!eS7S@>mI^P*P>DlFUw&u>ls+IA&kOl(Zt5@6wcB?iFygAC}WJrHW#8U56jnm)?xXo_9;xY2dzEg`MnbcLo;)OLxmLt28t7FP`E?yJE3-N7L*@c6pCYwzPVso&XeI`c)n(xVW&Z7R|#xYyWQ#0Obu>K5tW<!(j-Ct%6dL&!YUMAyCnxd~8UUPc{L^^wwaRxy0{6AhZf2>eRQJ!9plzdi4gN4ck}km*7#5G9qvY8TB1fPP&**RBYBQr`M>8~VXg7TN?gU6TisHL}G0tPtlQDe$lsEoS<~rfg<ZQ7vnddX&>`Ov;|^c_Ovx$;<ew@M`lRv30FIXlc>Ju-sjPm3)XyCX{&As`P&S|L<Vvoy04ojca!MO9vh>Dh5w15~*ub2Vy>tt&`a<1e0E|9R2hCe#OW`F*i1@WdtKb4wz}{nnrE=kiXVJqY45+BY6qGE9w-NoEQKOqj|itYt7F+N;`6H%De|ax^xycHqJC!=D*=wWSVfJoS>-^<k8MdLL!a(KAJk(`El~N46hWPyFS$Q`FRbv^lc?38wv)R9vL1>P4bI~Fa66z8st-zm?1cWY{0J{HRd$uf3AD;6ryE|XRRXpHnLrsm%Zp<m+aX$L%xxV2RTQd%UXfF08?TMyeg(%+vTr;wm%R=ktQ7))b^@$OWL7LsPs|n7N5@X?|$l&-f*fiN29M0pWS&oS3PU`d(`VICPPEHSz6WaTPLpX*UI{C5=#Nl1%7xvUml5?k6Kgl1XX<PGzozmBzYFJb4vqqP*;6sh|~<HF(=Wmo#TN8ZF^_sX63S152p+`8-!nBwxw6@F#x5Yn3Q`vjgErQ@vPlC{em1YAUPeRxP?u2b1kdvs^Efz>ZznH!>Veby9$_7#Uh38`VyEkG&5Ms40)+(Y`+kC1iygB2=1r<iTXXVAo&S-gKiZL2SZgiwr)|yw@2LaD{E7ya9go|XF@~!4}CG|t~d&LciAcfdZ2@oix^9XDVyk-b{dKLj8aIH?27yvIl6!}oJ*oVnNkn3Jw=DYtzUrn*+cuB2RQj$cExC?=$hZLQL3vnwWYBsvrvab6<)9*_7;I;i3w+qPoUS>GOJ}7&6Ai!hZfLpm@wS@_t{Xr{?UaB&D1_$FZeWIJ%o)!(MR^Kjj`Uir<F6uU4ClT6mdukQ`BKzyu9lM0b5Oj0b{~TQuQ)frKxTvkalQ>1)*aU1~1ee8jfK7q{e2l<1?CzmgVA$h>noA)<~*&qAfR`u%sRld@Y|I{>dEcGhxHnYPWCabQZLQ%AV|PlxAMN;G@;6Ic_m$A2nqk2#(Ho3TDgHRUU51?IITNe5Dsrw2-zH&(1Knydg_+++t<xq&q5D<xD5&{W3|KRM%Z7dLM%))~wZkZVWEeDZl4`TbDJ)09(-tjUej8i<E2n2)w|U*+iHIbLxNStQ!WhdInIW>RKdJWv)5!7~)5r-9LcaZ>U{l3vu6!TlFM+`<@P_PAG}$E^H3Bj<@@Y)NM3sI-WI5=~KBBu(SKRvSE$t`h4|<BZuTTC$_rWjl2&0E)a7dh#`uv3R9*&Q^$_sM?$$E7h~TJQS{59+61JDC~ucG)O=NdQ37?ps~f7svYE7gVbriQ5Hgw96VK#Hv(jhU);y;?dIv8LX!If_v3+5^qD9+~Y(=dRgzbG^4~2_=?qBx5XeU*#8u|TIM5H9E4k{YmVu(Ho=Mcq6qf;wqTK|HQS0qxmqUeIe9w;~7V~N&tEsH0}_aWzr2=c*d)UZQ(dOcQStQj3(I3_w$rUc*LU5c-gW66=gC*--P=4i25wNQAwAUOE}@V1}2-}*5sk*cmwuJ-#mt5-<~c}Hci4UqD;>?pXE>s_szoUU!1Fzk-wry%vNeKpjBDM)W8<{?^m(X40aq)4=05)F3Q<|~cbgu4*TU))`<GB&3S8AQm2QmB{}Vggq*p5I#L+)%JwS2Wvu14QoYbHlmGjgr8zma!Gye)_hs-oZ&rU+?=f<8?B0RX82^2gZbBJJk7agZiPQdcou!Bif8{<SfKv1(ycMyJkapu%PH7M(vRyG9%n0YLHi7L~&9!Sc-$EXDQY!VrMX#)X}DeeK1YG)!j~u(j<23?($!8c>ROMutv@{kw2fYFj^EK#8<7FJ`rqNSM~{h1x^%IkCsZR<xB14_QH!((%sdr#78fmMUysrj`!ed$;&cD?hO-%8Ti01>VucM1-P$FTHRwPQes=m92Nf|w0NDkMelX7k_QGUo4Ug2E;n$?#x2jdI@Ft=$XkblV?A#=qX|}Qu<Ss2)R$Jr^bpD%X!Y_qSkntd{t@p13a6iMXB!%y?ZkcM8SeUgNiy=rbuaktOLPPSE&j9cO>MU>GXi|PW;1M&wCTIF5({o+Zp-6jeYPAL8%~b(Fw@S(BDbbRx|%I;Ud;ZxWBFCdwB6C%d~r8{5lj=5Fshm5;WQpnD@WUk0Suk`&&JlLcxC6#tt>a@Vv(PMDKpO+x9O{iT{pMSyElbJF=NHh$N3(~kpJ1RHe^wO6M*h6TW!V+ceRoDJIL00Yab0xDET)s>5*50mn37Q+bu50Fax@Af2Hol8q@4N7U>DDSFeggV3G9!chnZ#j5uLTclR8Ug%h))7^tv|0|b$<#E>_p6ncam5SHJSM!?2k@;;Yrh0GM17fL!9O#rZxd)=IY={$;+qiz5Kc*qu-a-=N5XdDE7ZJ-2mwLwLufa(kElPyRO$eqGJ;Z}KR!!q!yEMB2@tDb4ZG`#F%De;rwiV{#c<e!aDQUO<kaqb?E(h-2hpW{$=1o=K5rviEZ;r!}3lQI!P7hWGgHlLg3QH<5X=nk-Ggb^SwKO%~}eYh(9k&?^I!EK{z`JC_l*3acCQN97|k6uo>3WTgbtPqEqyA1v3t%9&5fT&zA&7r+A>I%0z*L`naW|e@(j=cb;#)yFcYkFJaZc7UX`QUG$m6PcoAz)(_bO#P<t1#Ie1x$>C4rP9VWX&4^X*=X$e!l^u=KZAOJX7}l#DuaZw#VjRWP|14)vFO`$SJ?0E>*Z*vCXp`fkfxMKkTdt{6WXVpnXeB0a*05=4U)}R5IFtTU!cgPv6O`+Tek@jTb}JC2D7KL=4I-dSx*p0_K=*AKFA!HOTskGtNtae%EbivAS7loSVT&h!k&z!>HkCQToD{Dmd7l91%^>e6O}jmoo`53x4iA=lZX#L>~2Eq1|(g@tEE+w}NzRB@a(rIpbwnMd0^8hQ)`rN0x2uyu;#NL;s491|(Z=&4miFn}MV;ZU}g`>!PZuj-Uyi>{zz6>6O`yje!jE*lW1D4JLnC`MTs7qNMPs0K`@qdkRDm@pr*c&tar$>s$tO*%`=;4NOS`00FGzqLiYDIU-fCK>+<rK`5^G@D&vO$9(Pym-6)KD<^L?us~ixhRKPpvD?+d?Ux1mH5E3E=Js%!f8LjSQ^)|!Tb0-Ph_4x~rD4r~7eqj@>wxpc9DRtLCz6ZJ^t~lkY^!%Qa6LN!M049S&L6toC!auXWUp&I9zrBCTS_kmX}U*oR7J*kq{8`CZ{*q+5TLF1*xgbCXfmiJ+;AOD45c>+Pbu0P1q>~<)C9A+)7NSlIY`D!uS1s2km)_?`UJW4@e@{*M<-X2A_u<i#xiH*2e?^5wC8pR{x0A4#ZODoRntvpeyoJh0O#{v%?-#%7HVJ+=jpHT$z3-Ba*KH~;6d}r4$S<8yq^G|U+WujUEu7;or{;fT?(NPIw%2ENVo;zpQ&xK^5j?)k88Q5kl9j++^1+?J#S&EQ^QiYy=e~=B5i@g1`2&{^{Pyw%%y!LNfxoaj}#+GbaRO(?mRp?ge#=9iF#fJ;)$+IO~t&K!83G&qm6PTOXodgNo4lM18p$pfJ7@xXAa(l=1eHlD!P?YT_AGG@ZQPdwmo@-Y#lZ{5?IEvUbXI10fXlAJf@87gU35xm{(|@ft_h{1T#UHaoe_$JB5{uvABZshONdJ<@8$gVE+EiPETjN1BU6a2JSIRxq3Bk%Mng+y-*5JV1APHnCJ+H3A}ibhoW+P6nIgrBit713aBpLUVKf$j(Sm&gGH`IW=XV`*co-3sz`*!JN^FcosNgx%W6$}36AHu*I2<0M-MLOTawOAS|U?NVg@SkN#Ne^|MDm@k*&YGOc@!?4F>6g!Xmrt25rkhgg$hNr-GCqsRKw6#9R+S-$w-nBg;pd@xPbwE|&WFX%&?kyBO9A{wnov)YqF?CD7|zz}t^fmi#+X;!q9(*{}!nGD6Y!pUG2A=`EXBFJ~bJLlK8hChRAt{pv5T^t@;X;t(9!s9g~ONzJ_z6(vD6!>U3dr$T_pJ|-sJ`y_-@4Lb3P7~h9;0kaAGLxYnR+GM1Wrr>^pslfKd78DO$FB_KTp(51^O?L!0gLT4?+h%7l*Hr(?=0HHaObzwixj9&B(9i8C!Qu;thIq!RuVG1_F=n@D>UERV8Imp~G^WN_UJf0GgOmggYUJ6ZwIrHi=}bIlF+7LfC%ZGTp^DdsBmU4I0cw8qsEI-9*J`LZYA$|WC_EYs$$i!7gBz))bx<|yZPBoZ9cU%&XO+*AK@bKFeU9HtZhmeOt{`4$L)2=#@{3W-l7TBj*CXLAW8EyiV8Ybnk?o})A6K^SvLkb)j-Ej}2aBL}qAO11p0R~sz==GI+l(Fx6lwLfpBKudEEYyn3=C%0M6Le2hFwuqM*^UB>BiY*qnQEIwt~AVH%0y=2-o=Lr2?qxgT4D_@){Z^r7A@LZG;95ek)l~A4#<k)*MWkY+VV0M({@j{x##t&tASG1NM@QI$m1}<h*xK00000cR3d+X1RUo00H?du<8K-q;%y~vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
