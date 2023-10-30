DL project:

In this project we want to leverage the general-purpose audio modelling capabilities of neural audio codecs for speech enhancement. Because neural audio codecs are trained only to reconstruct their input, they can be (and are) trained on all kinds of diverse sound data.
In contrast, the (simple) speech enhancement pipeline involves training using paired (clean) speech and noise data sets, and then combining samples from the two to produce noisy speech, from which the network is then tasked with recovering the clean speech.
By starting from a pre-trained audio codec checkpoint, we can implicitly use sound data that is not constructed as an artificial combination of speech and noise (think e.g. a real-world recording of a speaker in noisy conditions), hopefully yielding a better result than just training from scratch using only paired data.
Fine-tuning a model from a pre-trained starting point is also typically much faster than training from scratch, which is super useful as we only have a few weeks.
We will start by borrowing some nice pre-trained audio codecs to use as starting points:
* https://github.com/facebookresearch/AudioDec
* https://github.com/descriptinc/descript-audio-codec
AudioDec has a causal model (i.e. zero algorithmic latency) so it may be a good starting point for those interested in real-time application, but Descript codec sounds better due to having been trained on substantially more data and having used more tricks for better audio modelling. If you aren't sure I recommend starting with the Descript codec.
