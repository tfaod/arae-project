from torchsummary import summary
from models1 import Seq2Seq2Decoder
from models3 import MLP_Classify, MLP_G, CNN_D


autoencoder = Seq2Seq2Decoder(emsize=256,
                              nhidden=256,                              
                              ntokens=30004,
                              nlayers=1,
                              noise_r=0.1,
                              hidden_init=False,
                              dropout=0.0,
                              gpu=None)


#gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_gen = MLP_G(ninput=32, noutput=256, layers='256-256')
# gan_disc = CNN_D(ninput=args.batch_size, noutput=1, activation="gelu")
gan_disc = CNN_D(ninput=64, noutput=1, activation="gelu")

# classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
classifier = MLP_Classify(ninput=256, noutput=1, layers="256-256")


# source shape is torch.Size([64, 25])
# lengths is an array of size 64

print(summary(autoencoder, (25)))
# gan_gen(torch.ones(args.batch_size, args.z_size)))
print(summary(gan_gen, (32)))
print(summary(gan_disc, (256)))

# output of autoencoder is input of classifier
# code = autoencoder(0, source, lengths)
# code shape is [64, 256]
print(summary(classifier, (22)))

