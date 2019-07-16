from model_class import SentVae
import numpy as np

zs = [[0] * 13,
      [1] * 13,
      [10] * 13,
      [100000] * 13,
      [1] + [0] * 12,
      [0] + [1] + [0] * 11,
      [0] * 2 + [1] + [0] * 10,
      [0] * 3 + [1] + [0] * 9,
      [0] * 4 + [1] + [0] * 8,
      [0] * 5 + [1] + [0] * 7,
      [0] * 6 + [1] + [0] * 6,
      [0] * 7 + [1] + [0] * 5,
      [0] * 8 + [1] + [0] * 4,
      [0] * 9 + [1] + [0] * 3,
      [0] * 10 + [1] + [0] * 2,
      [0] * 11 + [1] + [0] * 1,
      [0] * 12 + [1],
      [-1] * 13,
      [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
      list(range(13)),
      list(range(13)).reverse()]

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
        for sent in sents
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
            sent = vae.generate(z)
            log_file.write(str(z) + "\n")
            log_file.write(sent + "\n")
            log_file.write("\n\n")
    
