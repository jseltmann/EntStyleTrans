from model_class import SentVae
import numpy as np
import tensorflow as tf

zs = [[0] * 13,
      [1] * 13,
      [10] * 13,
      [100000] * 13,
      [1] + [0] * 12,
      [0] * 8 + [1] + [0] * 4,
      [0] * 12 + [1],
      [-1] * 13,
      [25] * 13,
      [48] * 6 + [-48] * 7,
      [-10000] * 13,
      [123,456,789] * 4 + [123]]

sents = ["ok never going back to this place again .",
         "easter day nothing open , heard about this place figured it would ok .",
         "the last couple years this place has been going down hill .",
         "why is the rum gone ?",
         "there was no water .",
         "colorless green ideas sleep furiously ."]



def test_vae(weights_filename, log_filename):
    vae = SentVae()
    vae.load_weights(weights_filename)

    inferred = []
    with open(log_filename, "w") as log_file:
        for sent in sents:
            mu, sigma = vae.inference(sent)
            z = vae.sample_z(mu, sigma)
            new_sent = vae.generate(z)
            log_file.write(sent + "\n")
            log_file.write(str(mu) + "   " + str(sigma) + "\n")
            log_file.write(str(z) + "\n")
            log_file.write(new_sent + "\n")
            log_file.write("\n\n")


def test_generator(weights_filename, log_filename):
    vae = SentVae()
    vae.load_weights(weights_filename)
    
    with open(log_filename, "w") as log_file:
        for z in zs:
            z = np.array(z)
            z = z.reshape((1,13))
            z = z.astype(np.float32)
            sent = vae.generate(z)
            log_file.write(str(z) + "\n")
            log_file.write(sent + "\n")
            log_file.write("\n\n")
    
#test_vae("model_weights/vae", "analyze_vae.log")
test_generator("model_weights/vae", "analyze_generator2.log")
