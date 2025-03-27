from sklearn.preprocessing import MultiLabelBinarizer

def decode_disc_labels(encoded_disc_labels):
    mlb = MultiLabelBinarizer(classes=['D', 'I', 'S', 'C'])
    mlb.fit([['D', 'I', 'S', 'C']])
    return mlb.inverse_transform(encoded_disc_labels)