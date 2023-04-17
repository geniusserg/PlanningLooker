import keras
import spacy
import pickle 
import os
from keras.utils import pad_sequences

##################################################
# Class which detects sentences with plans
# Input - directory with keras modela nd tokenizer
# Fit input - texts
# Output - boolean array 
##################################################
class PlansExtractor():
    SENTENCE_LENGTH = 20
    def __init__(self, model_path):
        self.model_w2v = keras.models.load_model(model_path) # model
        self.nlp = spacy.load("ru_core_news_lg") # lemmmatizer
        self.tokenizer = pickle.load(open(os.path.join(model_path, "tokenizer.pkl"), "br")) # tokenizer keras

    #return boolean
    def predict(self, input: list[str]):
        return self.predict_proba(input) > 0.5

    #return logits
    def predict_proba(self, input: list[str]):
        inputs = []
        for text in input:
            # to lemmas
            lemmas = []
            doc = self.nlp(text)
            for tok in doc:
                lemmas.append(tok.lemma_)
            sent = " ".join(lemmas)
            # to vector
            sequences = self.tokenizer.texts_to_sequences([sent])
            sent = pad_sequences(sequences, maxlen=PlansExtractor.SENTENCE_LENGTH)
            inputs.append([sent])
        # predict and mae binary labels from logits
        return self.model_w2v.predict(inputs) 


if __name__ == "__main__":
    model = PlansExtractor("model_cnn")
    print(model.predict(["Завтра я буду смотреть ну погоди утром!"]))

        
        
