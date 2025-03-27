import unittest
from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings
from utils.disc_labels import decode_disc_labels

class DISCPredictionTest(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    def test_disc_prediction(self):
        input_text = "\t\t<h1>It's nice to chat with you on 16/3/2025. \n\n Let's meet up next time!</h1>\n\n\t Dave"
        # input_text = "[IMAGE] [IMAGE] \t\n\n\n <h1>This is urgent, the deadline is tomorrow at 1:00pm! Finish it asap!</h1>\n\n\t Dave"
        cleaned_text = clean_msg_body(input_text)
        print(cleaned_text)

        embeddings = get_bert_embeddings([cleaned_text])
        print(embeddings)

        result = self.model.predict(embeddings)
        print(result)

        disc_labels = decode_disc_labels(result)
        print(disc_labels)

        acceptable_labels = [('D',), ('I',), ('S',), ('C',), ('D', 'I'), ('D', 'S'), ('D', 'C'), ('I', 'S'), ('I', 'C'), ('S', 'C'), ('D', 'I', 'S'), ('D', 'I', 'C'), ('D', 'S', 'C'), ('I', 'S', 'C'), ('D', 'I', 'S', 'C')]
        self.assertIn(disc_labels[0], acceptable_labels)

if __name__ == "__main__":
    unittest.main()