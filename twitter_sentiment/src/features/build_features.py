from transformers import BertTokenizer

# initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

index_to_class = {0: 'Negative', 1: 'Neutral'}


def preprocess(sentence):
    """
    Tokenizes a sentence and uses embedding indices as input for model predictions.

    Returns: tensor of token indices from sentence
    """

    # process inputs
    inputs = tokenizer(sentence, padding=True, return_tensors="pt")
    indices_tensor = inputs["input_ids"]

    return indices_tensor
