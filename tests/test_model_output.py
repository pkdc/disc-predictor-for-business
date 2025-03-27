import unittest
from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings

class DISCPredictionTest(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    def test_disc_prediction(self):
        input_text = "\t\t<h1>This is urgent, the deadline is tomorrow at 1:00pm! Finish it asap!</h1>\n\n\t Dave"

        cleaned_text = clean_msg_body(input_text)
        print(cleaned_text)

        embeddings = get_bert_embeddings([cleaned_text])
        print(embeddings)

        

if __name__ == "__main__":
    unittest.main()